# MJX GPU Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train the G1 field policy on a Kaggle GPU with thousands of parallel MJX environments, and load the result back into the existing CPU evaluation, rendering and watch scripts.

**Architecture:** A second env, `BipedMjxEnv` (`biped_rl/mjx_env.py`), is a MuJoCo Playground `MjxEnv` that copies `BipedEnv`'s observation, action mapping, reward and termination exactly, in JAX. `scripts/train_mjx.py` trains it with Brax PPO in curriculum stages (one `ppo.train` call per chunk, restoring params between chunks, gate checked by a deterministic eval rollout). The trained params are saved as `runs/<name>/latest.params`, and `scripts/evaluate.py:load_policy` learns to load them, so every existing script (evaluate, render_video, watch, progress_video's loader) runs Brax policies on the CPU `BipedEnv` unchanged.

**Tech Stack:** JAX (CUDA 12 on Kaggle, CPU locally), `mujoco` + `mujoco-mjx` (same version), `playground` (MuJoCo Playground), `brax` (PPO), `ml_collections`.

**Spec:** No separate spec. The source of truth is `biped_rl/env.py` (behaviour to copy) and `scripts/train.py` (curriculum levels and gate). README "Stage 2 (MJX/Playground on Kaggle)" is the motivation.

## Global Constraints

- NEVER run `git add`, `git commit`, `git push` (user's CLAUDE.md). Every "Commit" step below is replaced by: leave changes uncommitted, list changed files in the task report. Subagents must be told this.
- Scope is the **field scene, flat terrain, `LEVELS_FIELD` levels 0-4** (stand, walk 0.3 / 0.6 / 1.0 m/s, pushes up to 10 N·s). Roughness (levels 5+), the course scene and `--height-scan` are out of scope; `BipedMjxEnv.__init__` raises `NotImplementedError` for them.
- Constants come from `biped_rl.env` by import (`LEG_Q_DEF`, `LEG_ACT_SCALE`, `N_LEG`, `FALL_HEIGHT`, `FULL_HEIGHT`, `AIR_TIME_TARGET`, `MIN_GAIT_SPEED`, `MIN_CLEARANCE`, `DEFAULT_REWARD_SCALES`, `SCENES`, body/geom names). No copies.
- Observation layout must equal `BipedEnv._obs()` (50 floats, same order, `dqj * 0.1`, `min(dist,10)/10`). This is what lets a GPU-trained policy run in the CPU env.
- Policy rate 50 Hz: `opt.timestep` 0.004, 5 substeps per policy step, `DT = 0.02`.
- Network sizes match the CPU run: policy (256, 256), value (256, 256). Discount 0.99, GAE lambda 0.95, clip 0.2.
- Curriculum gate identical to `CurriculumCallback`: survival ≥ 0.9 and, when speed > 0, mean `path / (len * DT)` ≥ 0.7 × speed.
- GPU-only deps go in a new `requirements-gpu.txt`; `requirements.txt` stays CPU-only. Tests that need JAX start with `pytest.importorskip("mujoco.mjx")` and `pytest.importorskip("mujoco_playground")` so the CPU suite stays green without them.
- Pin exact versions in `requirements-gpu.txt` to what Task 1 resolves; `mujoco` and `mujoco-mjx` must be the same version.

## Review Focus

1. A NaN/exploded MJX env (bad contact, huge velocity) must terminate as a fall, not poison the batch: `done = 1` and the reward is finite when `qpos` has a non-finite entry. Test in Task 3.
2. Goal resampling near the field edge must stay inside |x|,|y| < 12 (CPU retries 20 times; MJX falls back to the origin direction). Test in Task 3.
3. `kp_scale` must scale both `actuator_gainprm[:,0]` and `actuator_biasprm[:,1]` on the model MJX actually uses, including after domain randomization. Test in Task 4.
4. A `.params` file must load and act on a machine with CPU-only JAX (Kaggle-trained, laptop-evaluated). Test in Task 6.
5. Gate math with an empty or zero-length eval batch must not divide by zero and must not pass the gate. Test in Task 5.

---

### Task 1: GPU dependencies and MJX smoke check

**Files:**
- Create: `requirements-gpu.txt`
- Test: `tests/test_mjx_env.py`

**Interfaces:**
- Produces: installed `jax`, `mujoco-mjx`, `playground`, `brax`; the fact (recorded in the ledger) of whether MJX collides `foot_capsule` geoms with the `terrain` hfield in `assets/g1/scene_g1.xml`.

- [ ] **Step 1: Write `requirements-gpu.txt`**

```
-r requirements.txt
jax[cuda12]        # on Kaggle; locally `pip install jax` (CPU) is enough for tests
mujoco-mjx==<same as mujoco>
playground
brax
ml_collections
```
Install locally into the venv with CPU jax: `venv/Scripts/python -m pip install jax mujoco-mjx==<mujoco version> playground brax ml_collections`. Replace `<...>` and add `==` pins for every line with the versions pip resolved (`pip show`).

- [ ] **Step 2: Write the failing smoke test**

```python
mjx = pytest.importorskip("mujoco.mjx")

def test_mjx_feet_touch_hfield_at_home_pose():
    m = mujoco.MjModel.from_xml_path(SCENES["field"])
    cpu = BipedEnv(randomize=False); cpu.reset(seed=0)   # places feet on the flat hfield
    d = mjx.put_data(m, cpu.data)
    d = jax.jit(mjx.step)(mjx.put_model(m), d)
    assert np.isfinite(np.asarray(d.qpos)).all()
    assert foot_contacts(m, d).tolist() == [1.0, 1.0]
```
`foot_contacts` is the Task 2 function; until it exists the test fails on import.

- [ ] **Step 3: Run it**

Run: `venv/Scripts/python -m pytest tests/test_mjx_env.py -v`
Expected: FAIL on `ImportError: foot_contacts` (not on `put_model`). If `put_model`/`step` itself raises "not implemented" for hfield×capsule, ledger a Ruling: create `assets/g1/scene_g1_flat.xml` = `scene_g1.xml` with the `terrain` geom changed to `type="plane" size="15 15 0.1" pos="0 0 0.25"` (0.25 = flat hfield height 0.5 × size[2] 0.5), and use it in `BipedMjxEnv`. Cost if wrong: roughness later needs the hfield path.

---

### Task 2: `BipedMjxEnv` reset and observation (parity with CPU)

**Files:**
- Create: `biped_rl/mjx_env.py`
- Test: `tests/test_mjx_env.py`

**Interfaces:**
- Consumes: constants from `biped_rl.env`.
- Produces:
  - `foot_contacts(mj_model, data) -> jax.Array (2,)` — 1.0 if any of that foot's 3 collision geoms collides with `terrain` (`mujoco_playground._src.collision.geoms_colliding`).
  - `default_config() -> ml_collections.ConfigDict` with keys `kp_scale=1.0, target_speed=1.0, push_max=0.0, push_interval=(2.0, 5.0), push_duration=0.1, target_range=(4.0, 8.0), target_angle=0.6, reach_radius=0.5, randomize=True, reward_scales=dict(DEFAULT_REWARD_SCALES), episode_length=1000, ctrl_dt=0.02, sim_dt=0.004`.
  - `class BipedMjxEnv(mujoco_playground.MjxEnv)`: `__init__(config=None, config_overrides=None)`; `reset(rng) -> State`; `step(state, action) -> State`; properties `xml_path`, `action_size == 12`, `mj_model`, `mjx_model`; `obs(data, info) -> jax.Array (50,)`.
  - `State.info` keys (used by Tasks 3-5 and by tests): `rng, target (2,), prev_action (12,), air_time (2,), was_in_contact (2,), prev_contacts (2,), foot_xy (2,2), steps, path, n_reached, next_push, push_end, push_vec (3,)`.
  - `State.metrics` keys: `fell, path, n_reached`.

- [ ] **Step 1: Write the failing parity test**

```python
def test_obs_matches_cpu_env_at_same_state():
    cpu = BipedEnv(randomize=False, kp_scale=3.0); cpu.reset(seed=0)
    for _ in range(10): cpu.step(cpu.action_space.sample())
    env = BipedMjxEnv(config_overrides=dict(kp_scale=3.0, randomize=False))
    data = mjx.put_data(env.mj_model, cpu.data)
    data = mjx.forward(env.mjx_model, data)
    info = dict(target=cpu._target, prev_action=cpu._prev_action)   # obs reads only these two
    np.testing.assert_allclose(env.obs(data, info), cpu._obs(), atol=1e-4)

def test_reset_shapes_and_jit():
    env = BipedMjxEnv(config_overrides=dict(randomize=False))
    s = jax.jit(env.reset)(jax.random.PRNGKey(0))
    assert s.obs.shape == (50,) and float(s.done) == 0.0
    assert np.isfinite(np.asarray(s.obs)).all()```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: biped_rl.mjx_env`).

- [ ] **Step 3: Implement.** `__init__`: load `SCENES["field"]` (or the Task 1 flat XML), apply `kp_scale` to the `mj_model` before `mjx.put_model`, store `home_qpos/home_ctrl/qadr/dadr/jrange` exactly as `BipedEnv.__init__` computes them, and the flat ground height `ground_z = 0.5 * hfield_size[0, 2]`. `reset(rng)`: home pose, ±0.03 joint noise and ±0.05 qvel noise when `randomize`, lift so the lowest foot point is 5 mm above `ground_z` (compute the lift once in `__init__` with CPU `mj_forward` on the home pose — noise is ≤ 3 mm, ponytail: recompute per reset if feet start inside the floor), sample target with `_sample_target`. `obs` follows `BipedEnv._obs` term by term.

- [ ] **Step 4: Run, expect PASS** for both tests and the Task 1 smoke test.

- [ ] **Step 5: Leave uncommitted** (Global Constraints).

---

### Task 3: `step`: physics, reward, termination, goal resampling

**Files:**
- Modify: `biped_rl/mjx_env.py`
- Test: `tests/test_mjx_env.py`

**Interfaces:**
- Produces: `BipedMjxEnv.step(state, action) -> State` with `state.reward` = the CPU reward formula, `state.done` = terminated (fell or NaN); truncation at `episode_length` is left to Playground's episode wrapper. `_sample_target(rng, base_xy) -> (2,)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_step_reward_close_to_cpu():
    cpu = BipedEnv(randomize=False, kp_scale=3.0, target_speed=0.6); cpu.reset(seed=1)
    env = BipedMjxEnv(config_overrides=dict(kp_scale=3.0, target_speed=0.6, randomize=False))
    s = state_from_cpu(env, cpu)          # test helper in this file: put_data + info from cpu._* fields
    a = np.full(12, 0.2)
    _, r_cpu, term, _, _ = cpu.step(a)
    s2 = jax.jit(env.step)(s, jnp.asarray(a))
    assert abs(float(s2.reward) - r_cpu) < 0.05 and bool(s2.done) == term
    np.testing.assert_allclose(s2.obs, cpu._obs(), atol=5e-2)

def test_nan_state_terminates_as_fall():
    env = BipedMjxEnv(config_overrides=dict(randomize=False))
    s = env.reset(jax.random.PRNGKey(0))
    s = s.replace(data=s.data.replace(qvel=s.data.qvel.at[0].set(jnp.nan)))
    s2 = env.step(s, jnp.zeros(12))
    assert float(s2.done) == 1.0 and float(s2.metrics["fell"]) == 1.0
    assert np.isfinite(float(s2.reward))

def test_resampled_target_stays_in_bounds():
    env = BipedMjxEnv()
    keys = jax.random.split(jax.random.PRNGKey(0), 1000)
    t = jax.vmap(env._sample_target, in_axes=(0, None))(keys, jnp.array([11.5, 11.5]))
    assert float(jnp.abs(t).max()) < 12.0

def test_standing_still_survives_one_second():
    env = BipedMjxEnv(config_overrides=dict(kp_scale=4.0, randomize=False, target_speed=0.0))
    s = env.reset(jax.random.PRNGKey(0)); step = jax.jit(env.step)
    for _ in range(50): s = step(s, jnp.zeros(12))
    assert float(s.done) == 0.0
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** following `BipedEnv.step` line for line in `jnp`: ctrl clip, 5 substeps (`jax.lax.fori_loop` over `mjx.step`), path, up_z, foot height, clearance against `ground_z`, out-of-bounds (15 m field), reward terms with `self._config.reward_scales`, air-time / slip with the info state, `on_reach="resample"` via `jnp.where(reached, _sample_target(...), target)` (+`reach` reward, `n_reached += 1`), fall = `up_z<0.5 | height<FALL_HEIGHT | clearance<MIN_CLEARANCE | oob | ~isfinite(qpos).all() | ~isfinite(qvel).all()`, reward `-= fall_scale` on fall, then `reward = jnp.nan_to_num(reward)` and `obs = jnp.nan_to_num(obs)`. `_sample_target`: one draw of angle/dist; if any |coord| ≥ 12 return the point at distance `target_range[0]` toward the origin. Energy is not computed (CPU-only eval metric).

- [ ] **Step 4: Run, expect PASS** (all tests in the file).

- [ ] **Step 5: Leave uncommitted.**

---

### Task 4: Pushes and domain randomization

**Files:**
- Modify: `biped_rl/mjx_env.py`
- Test: `tests/test_mjx_env.py`

**Interfaces:**
- Produces: push logic inside `step`; `domain_randomize(mjx_model, rng: (B,2)) -> (batched_model, in_axes)` for `wrap_for_brax_training(randomization_fn=...)`.

- [ ] **Step 1: Write the failing tests**

```python
def test_push_applied_for_push_duration():
    env = BipedMjxEnv(config_overrides=dict(randomize=False, push_max=10.0, push_interval=(0.04, 0.04)))
    s = env.reset(jax.random.PRNGKey(3)); step = jax.jit(env.step)
    forces = []
    for _ in range(10):
        s = step(s, jnp.zeros(12)); forces.append(float(jnp.abs(s.data.xfrc_applied).sum()))
    on = sum(f > 0 for f in forces)
    assert 1 <= on <= 6        # 0.1 s push = 5 policy steps

def test_domain_randomize_keeps_kp_scale_and_varies_mass():
    env = BipedMjxEnv(config_overrides=dict(kp_scale=3.0))
    m, axes = domain_randomize(env.mjx_model, jax.random.split(jax.random.PRNGKey(0), 4))
    assert m.body_mass.shape[0] == 4 and float(jnp.std(m.body_mass[:, 1])) > 0
    np.testing.assert_allclose(m.actuator_gainprm[0, :, 0], env.mj_model.actuator_gainprm[:, 0], rtol=1e-6)
    assert np.allclose(env.mj_model.actuator_gainprm[:, 0], 3.0 * mujoco.MjModel.from_xml_path(SCENES["field"]).actuator_gainprm[:, 0])
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** Push: checked once per policy step (`t = steps * DT`); start when `t >= next_push`, impulse `U(0, push_max)`, angle `U(-π, π)`, force = impulse / `push_duration` on the pelvis `xfrc_applied[:3]`, cleared at `push_end`, `next_push = t + U(push_interval)`. Ruling to ledger: per-policy-step push timing instead of per-substep — 0.1 s is exactly 5 policy steps, same impulse. `domain_randomize`: per env one mass scale `U(0.85, 1.15)` on `body_mass` and `body_inertia`, one friction scale `U(0.7, 1.3)` on `pair_friction[:, :2]`; `in_axes` is `mjx_model.tree_replace` with 0 on those three fields, `None` elsewhere (Playground's `G1` randomize.py is the reference pattern). `randomize=False` disables reset noise only; randomization is chosen in `train_mjx.py`.

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Leave uncommitted.**

---

### Task 5: `scripts/train_mjx.py` — staged curriculum with Brax PPO

**Files:**
- Create: `scripts/train_mjx.py`
- Test: `tests/test_train_mjx.py`

**Interfaces:**
- Consumes: `BipedMjxEnv`, `default_config`, `domain_randomize`; `LEVELS_FIELD`, `DT` from `train` (scripts on pytest path).
- Produces:
  - `gate_stats(fell, path, length) -> dict(survival, track)` — numpy arrays per eval episode; empty or all-zero `length` → `survival=0.0, track=0.0`.
  - `gate_passed(stats, speed, survive_threshold=0.9, track_ratio=0.7) -> bool`.
  - `eval_rollout(env, inference_fn, rng, n_envs=256, steps=1000) -> (fell, path, length)` — deterministic, jitted, vmapped, stops counting an env after its first `done`.
  - Files in `runs/<name>/`: `env_kwargs.json` (`{"backend": "mjx", "kp_scale": ..., "scene": "field", "height_scan": false}`), `stage_<k>_<steps>.params` after each chunk, `latest.params`, `progress.csv` (`steps,level,speed,survival,track,eval_reward`).
  - CLI: `--name`, `--kp-scale` 1.0, `--start-level` 0, `--max-level` 4, `--chunk-steps` 20_000_000, `--total-steps` 200_000_000, `--num-envs` 2048, `--resume <path.params>`, `--seed` 0, `--no-randomize`.

- [ ] **Step 1: Write the failing tests**

```python
def test_gate_stats_and_pass():
    s = gate_stats(np.array([0, 0, 1, 0]), np.array([10., 10., 1., 10.]), np.array([1000, 1000, 50, 1000]))
    assert s["survival"] == 0.75 and abs(s["track"] - np.mean([0.5, 0.5, 1/(50*DT), 0.5])) < 1e-9
    assert not gate_passed(s, 0.6)
    assert gate_passed(dict(survival=0.95, track=0.5), 0.6)
    assert gate_passed(dict(survival=0.95, track=0.0), 0.0)

def test_gate_empty_batch_does_not_pass():
    s = gate_stats(np.array([]), np.array([]), np.array([]))
    assert s == dict(survival=0.0, track=0.0) and not gate_passed(s, 0.0)
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement.** Loop: build `BipedMjxEnv` for the current level (`push_max, _, speed = LEVELS_FIELD[level]`), call `brax.training.agents.ppo.train.train(environment=env, num_timesteps=chunk, num_envs, episode_length=1000, action_repeat=1, unroll_length=20, batch_size=256, num_minibatches=32, num_updates_per_batch=4, learning_rate=3e-4, entropy_cost=0.005, discount=0.99, gae_lambda=0.95, clipping_epsilon=0.2, normalize_observations=True, reward_scaling=1.0, network_factory=partial(ppo_networks.make_ppo_networks, policy_hidden_layer_sizes=(256,256), value_hidden_layer_sizes=(256,256)), wrap_env_fn=wrapper.wrap_for_brax_training, randomization_fn=None if --no-randomize else domain_randomize, restore_params=params, progress_fn=print_row, seed)`; then `eval_rollout` → `gate_stats` → append `progress.csv` → save params with `brax.io.model.save_params` → advance level if `gate_passed` and `level < max_level`; stop at `total_steps`. If the installed brax has no `restore_params` argument, ledger a Ruling and use `save_checkpoint_path`/`restore_checkpoint_path` instead.

- [ ] **Step 4: Run tests, expect PASS.** Then a CPU smoke run:

Run: `venv/Scripts/python scripts/train_mjx.py --name mjx_smoke --num-envs 8 --chunk-steps 40960 --total-steps 81920 --no-randomize`
Expected: two `progress.csv` rows, `runs/mjx_smoke/latest.params` exists, no NaN in printed reward. (Slow on CPU because of JIT; minutes, not seconds.)

- [ ] **Step 5: Leave uncommitted;** delete `runs/mjx_smoke` after Task 6 uses it.

---

### Task 6: Load `.params` policies in the CPU scripts

**Files:**
- Modify: `scripts/evaluate.py:46-56` (`load_policy`)
- Test: `tests/test_mjx_policy.py`

**Interfaces:**
- Consumes: `latest.params` + `env_kwargs.json` from Task 5.
- Produces: `load_policy(model_path, stats_path)` — when `model_path` ends with `.params`, returns `act(obs: np.ndarray (50,)) -> np.ndarray (12,)` built from `ppo_networks.make_inference_fn(make_ppo_networks(50, 12, ...))(params, deterministic=True)`, jitted on CPU; `stats_path` is only used to find `env_kwargs.json` (pass the same `.params` path). SB3 path unchanged.

- [ ] **Step 1: Write the failing test**

```python
def test_params_policy_acts_in_cpu_env(tmp_path):
    path = make_random_params(tmp_path)   # helper: init ppo networks (obs 50, act 12, 256x256), save_params, write env_kwargs.json
    act = load_policy(path, path)
    env = BipedEnv(**load_env_kwargs(path)); obs, _ = env.reset(seed=0)
    a = act(obs)
    assert a.shape == (12,) and np.all(np.abs(a) <= 1.0)
    env.step(a)
```

- [ ] **Step 2: Run, expect FAIL** (SB3 `PPO.load` on a `.params` file).

- [ ] **Step 3: Implement** the `.params` branch; import brax/jax inside the branch so CPU-only installs without brax keep working.

- [ ] **Step 4: Run, expect PASS,** then: `venv/Scripts/python scripts/render_video.py --model runs/mjx_smoke/latest.params --stats runs/mjx_smoke/latest.params --out outputs/videos/mjx_smoke.mp4`
Expected: an MP4 is written (the policy is untrained; it falls — that's fine).

- [ ] **Step 5: Full suite:** `venv/Scripts/python -m pytest -q`. Expected: all pass. Leave uncommitted.

---

### Task 7: Kaggle notebook and README

**Files:**
- Create: `kaggle/train_mjx_kaggle.ipynb`
- Modify: `README.md` (section "Training on a GPU (MJX)"), `.gitignore` nothing new

**Interfaces:**
- Consumes: `requirements-gpu.txt`, `scripts/train_mjx.py` CLI.

- [ ] **Step 1: Notebook cells** (same style as `kaggle/train_kaggle.ipynb`): locate dataset + copy to `/kaggle/working/biped_rl`; `pip install -q -r requirements-gpu.txt && pip install -q -e . --no-deps`; `import jax; print(jax.devices())` must show a CUDA device (assert it); set `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `MUJOCO_GL=egl`; run `python scripts/train_mjx.py --name mjx_walk --kp-scale 3 --num-envs 2048 --total-steps 200000000 | tee runs/mjx_walk/train.log`; zip `runs/mjx_walk` to Output. Accelerator: GPU T4 x2 or P100 (uses one GPU).

- [ ] **Step 2: README section:** what it trains (levels 0-4 only), the Kaggle steps, how to evaluate locally: `python scripts/evaluate.py --model runs/mjx_walk/latest.params --stats runs/mjx_walk/latest.params --label mjx --out outputs/eval/mjx` (needs `pip install jax brax` in the local venv), and that SB3 checkpoints and `.params` are not interchangeable.

- [ ] **Step 3: Measure on Kaggle (user runs it):** record steps/s from the first `progress_fn` rows and wall time per 20M chunk in the README. Expected: well above the CPU's ~1,000 steps/s; if below 10,000, lower `--num-envs` to 1024 and check `jax.devices()`.

- [ ] **Step 4: Leave uncommitted;** report all changed files.
