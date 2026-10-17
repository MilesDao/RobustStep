# Stand-Then-Walk Reward & Speed Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Get the G1 policy out of the "lunge forward and fall after ~1.6 s" local optimum by teaching it to stand first, then walk, with gait-shaping reward terms.

**Architecture:** Make the commanded walking speed a curriculum variable (starts at 0 m/s, so standing still earns the full velocity reward), raise the alive bonus, and add two standard gait terms (feet air time, foot slip). Reward computation is pulled out of `step()` into small methods so each term is unit-testable. Observation layout is unchanged (50 / 85 dims), so `evaluate.py`, `render_video.py`, `watch.py` and old checkpoints keep working.

**Tech Stack:** Python, MuJoCo ≥3.2, Gymnasium ≥1.0, Stable-Baselines3 ≥2.3 (2.9.0 in `venv/`), pytest.

**Spec:** No separate spec file. The design comes from the review on 2026-10-17, summarized in "Background" below; executors read that section as the spec.

## Background (the spec)

Observed on run `run1` at ~1.0M steps (fps ≈ 1340): `ep_len_mean` flat at 77–84 steps (~1.6 s), `ep_rew_mean` ≈ 67 (≈ 0.83/step), `curriculum/survival_recent = 0`, curriculum stuck at level 0, `explained_variance ≈ 0.98`.

Diagnosis (`biped_env.py:440-450`):
1. `target_speed = 1.0` from step 0 and `vel = exp(-4·(1.0 − v)²)`. Standing still earns `exp(-4) ≈ 0.018`; falling toward the goal briefly earns ≈ 1.0. Lunging dominates.
2. `alive = 0.1` vs `fall = 5.0`: almost no incentive to survive longer.
3. No gait terms: even once it survives, nothing discourages shuffling or sliding feet.
4. The curriculum gate (90% no-fall) is unreachable while the robot cannot stand, so push/terrain levels never start.

Required changes:
- R1. Commanded speed is a curriculum variable, starting at 0.0 m/s.
- R2. `alive` default raised to 0.5.
- R3. New term `air_time` (reward, default 1.0): on each foot touchdown, add `(air_time − 0.4 s)`; zero when commanded speed < 0.1 m/s.
- R4. New term `slip` (penalty, default 0.1): sum over feet in contact of squared horizontal foot velocity.
- R5. Field curriculum: stand → walk 0.3 → 0.6 → 1.0 m/s on flat ground, then the existing push/roughness levels at 1.0 m/s.
- R6. A 5M-step experiment with explicit pass/fail criteria decides whether the reward fix works or whether `--kp-scale` (control authority) is the bottleneck.

## Global Constraints

- Observation dimension stays exactly 50 (85 with `--height-scan`); no new observation entries.
- `BipedEnv(...)` default `target_speed` stays `1.0`, so evaluation scripts keep measuring walking at 1.0 m/s without changes.
- Policy rate unchanged: 50 Hz, `dt = 0.02 s` (timestep 0.004 s, frame_skip 5).
- Every reward term is a key in `DEFAULT_REWARD_SCALES` and overridable via `--scale name=value` (ablations rely on this).
- No new runtime dependencies; `pytest` is added to `requirements.txt` as a dev tool only.
- Comments and strings in English, matching existing style (short inline `#` comments, one-line docstrings).
- Run everything with the project venv: `venv\Scripts\python` (Windows).

## Review Focus

1. Resuming an old run (`--resume runs/run1/latest.zip`) after this change: obs shape must still be 50 so PPO.load works → `test_obs_shape_unchanged` (Task 1).
2. `evaluate.py` / `render_video.py` / `watch.py` build `BipedEnv` without a speed argument: default must stay 1.0 m/s → `test_default_target_speed_is_one` (Task 1).
3. `LEVELS_COURSE` and `LEVELS_FIELD` entries must all unpack as `(push, terrain, speed)`; a 2-tuple left anywhere crashes `_apply` mid-training → `test_levels_are_triples` (Task 3).
4. First `step()` after `reset()`: foot-velocity finite difference must not use stale positions from the previous episode (would produce a huge slip penalty) → `test_first_step_reward_finite_and_small_slip` (Task 2).
5. Ablation `--scale air_time=0` / `--scale slip=0` must be accepted, not raise "invalid reward_scales" → `test_new_scales_overridable` (Task 2).

---

## File Structure

- Modify `biped_env.py`: reward defaults, `set_difficulty` signature, reward helper methods, per-foot state reset, `step()` uses helpers.
- Modify `train_sb3.py`: level tuples gain a speed, `CurriculumCallback._apply` passes it and logs it.
- Modify `README.md`: reward term list, curriculum description, a note on the new defaults.
- Modify `requirements.txt`: add `pytest`.
- Create `tests/test_reward.py`: env-level reward tests (Tasks 1–2).
- Create `tests/test_curriculum.py`: curriculum tests (Task 3).
- Create `.gitignore`: `venv/`, `runs/`, `eval/`, `previews/`, `__pycache__/`, `*.mp4`, `*.gif`.

---

### Task 0: Put the project under git

The folder is not a git repository; every later task commits.

- [ ] **Step 1:** Create `.gitignore` with the entries listed in File Structure.
- [ ] **Step 2:** Run `git init` then `git add -A` then `git status`. Expected: no file under `venv/` or `runs/` is staged.
- [ ] **Step 3:** `git commit -m "chore: baseline before reward rework"`

---

### Task 1: Speed as a curriculum variable + alive bonus (R1, R2)

**Files:**
- Modify: `biped_env.py:54-66` (`DEFAULT_REWARD_SCALES`), `biped_env.py:176-185` (`set_difficulty`)
- Modify: `requirements.txt`
- Test: `tests/test_reward.py`

**Interfaces:**
- Produces: `BipedEnv.set_difficulty(self, push_max: float, terrain: float, speed: float | None = None) -> None`. `speed=None` leaves `self.target_speed` unchanged. Task 3 calls it with three positional args through `env_method`.
- Produces: `DEFAULT_REWARD_SCALES["alive"] == 0.5`.

- [ ] **Step 1: Add `pytest` to `requirements.txt`, then `venv\Scripts\python -m pip install pytest`.**

- [ ] **Step 2: Write the failing tests in `tests/test_reward.py`**

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from biped_env import BipedEnv, DEFAULT_REWARD_SCALES

def make(**kw):
    env = BipedEnv(randomize=False, **kw)
    env.reset(seed=0)
    return env

def test_default_target_speed_is_one():
    assert make().target_speed == 1.0

def test_obs_shape_unchanged():
    assert make().observation_space.shape == (50,)
    assert make(height_scan=True).observation_space.shape == (85,)

def test_set_difficulty_speed():
    env = make()
    env.set_difficulty(10.0, 0.02, 0.3)
    assert (env.push_max, env.roughness, env.target_speed) == (10.0, 0.02, 0.3)
    env.set_difficulty(0.0, 0.0)          # speed omitted -> unchanged
    assert env.target_speed == 0.3

def test_alive_default():
    assert DEFAULT_REWARD_SCALES["alive"] == 0.5
```

- [ ] **Step 3: Run** `venv\Scripts\python -m pytest tests/test_reward.py -v`. Expected: `test_set_difficulty_speed` fails with `TypeError` (too many positional args), `test_alive_default` fails (0.1 != 0.5); the other two pass.

- [ ] **Step 4: Implement.** Add the optional `speed` parameter to `set_difficulty` (update its docstring: "`speed`: commanded walking speed (m/s); None keeps the current one."). Change `alive=0.1` to `alive=0.5` in `DEFAULT_REWARD_SCALES`.

- [ ] **Step 5: Run** the same command. Expected: 4 passed.

- [ ] **Step 6: Commit** `git add biped_env.py requirements.txt tests/test_reward.py` then `git commit -m "feat: commanded speed as curriculum variable, alive bonus 0.5"`

---

### Task 2: Reward helpers, feet air time, foot slip (R3, R4)

**Files:**
- Modify: `biped_env.py` (`DEFAULT_REWARD_SCALES`, `reset()` around line 327-334, `step()` lines 409-450)
- Test: `tests/test_reward.py`

**Interfaces:**
- Consumes: `self.target_speed` (Task 1), `self._foot_contacts() -> np.ndarray` shape (2,) of 0/1 (existing, `biped_env.py:243`), `self.foot_bodies` (existing).
- Produces:
  - `DEFAULT_REWARD_SCALES["air_time"] = 1.0`, `["slip"] = 0.1` (comments: `# reward per touchdown for steps longer than 0.4 s`, `# penalty on feet sliding while in contact`).
  - Module constant `AIR_TIME_TARGET = 0.4` (s) and `MIN_GAIT_SPEED = 0.1` (m/s), next to `FALL_HEIGHT`.
  - `BipedEnv._air_time_reward(self, contacts: np.ndarray) -> float` — unscaled. Updates `self._air_time` (shape (2,), s). Algorithm: `first = (contacts > 0) & (self._air_time > 0)`; `r = sum((self._air_time - AIR_TIME_TARGET) * first)`; then `self._air_time = where(contacts > 0, 0, self._air_time + self.dt)`. Return `0.0` (but still update state) when `self.target_speed < MIN_GAIT_SPEED`.
  - `BipedEnv._slip_penalty(self, contacts: np.ndarray, foot_vel_xy: np.ndarray) -> float` — unscaled, `sum(contacts * sum(foot_vel_xy**2, axis=1))`; `foot_vel_xy` shape (2, 2).
  - `reset()` sets `self._air_time = np.zeros(2)` and `self._foot_xy = np.array([d.xpos[b][:2] for b in self.foot_bodies])` after the final `mj_forward`.
  - `step()`: before the frame-skip loop nothing changes; after it compute `foot_xy` the same way, `foot_vel_xy = (foot_xy - self._foot_xy) / self.dt`, store `self._foot_xy = foot_xy`, `contacts = self._foot_contacts()`, then add `+ sc["air_time"] * self._air_time_reward(contacts) - sc["slip"] * self._slip_penalty(contacts, foot_vel_xy)` to the existing sum. Put `"air_time": ...` and `"slip": ...` (unscaled values) into `info` for debugging.

- [ ] **Step 1: Append failing tests to `tests/test_reward.py`**

```python
def test_air_time_rewards_long_steps_only_when_walking():
    env = make()
    env.target_speed = 1.0
    env._air_time = np.zeros(2)
    for _ in range(25):                                   # left foot airborne 0.5 s
        assert env._air_time_reward(np.array([0.0, 1.0])) == 0.0
    assert np.isclose(env._air_time_reward(np.array([1.0, 1.0])), 0.5 - 0.4)
    assert env._air_time_reward(np.array([1.0, 1.0])) == 0.0   # no new touchdown

def test_air_time_zero_when_standing():
    env = make()
    env.target_speed = 0.0
    env._air_time = np.zeros(2)
    for _ in range(25):
        env._air_time_reward(np.array([0.0, 1.0]))
    assert env._air_time_reward(np.array([1.0, 1.0])) == 0.0

def test_slip_only_counts_feet_in_contact():
    env = make()
    v = np.array([[1.0, 0.0], [0.0, 2.0]])
    assert env._slip_penalty(np.array([1.0, 0.0]), v) == 1.0
    assert env._slip_penalty(np.array([1.0, 1.0]), v) == 5.0
    assert env._slip_penalty(np.array([0.0, 0.0]), v) == 0.0

def test_first_step_reward_finite_and_small_slip():
    env = make()
    for _ in range(3):                                    # episodes far apart in xy must not leak
        env.reset(seed=1)
        _, r, *_ , info = env.step(np.zeros(12))
        assert np.isfinite(r) and info["slip"] < 1.0

def test_new_scales_overridable():
    BipedEnv(randomize=False, reward_scales={"air_time": 0.0, "slip": 0.0})

def test_random_actions_stay_finite():
    env, rng = make(), np.random.default_rng(0)
    for _ in range(300):
        obs, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 12))
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        if term or trunc:
            env.reset()
```

- [ ] **Step 2: Run** `venv\Scripts\python -m pytest tests/test_reward.py -v`. Expected: the six new tests fail with `AttributeError` / `KeyError` / `ValueError: invalid reward_scales`.

- [ ] **Step 3: Implement** everything in this task's Produces block.

- [ ] **Step 4: Run** the same command. Expected: 10 passed. Then `venv\Scripts\python check_env.py`; expected: completes, `NaN=False`.

- [ ] **Step 5: Commit** `git commit -am "feat: feet air time and foot slip reward terms"` (also `git add tests/test_reward.py`).

---

### Task 3: Speed curriculum in training (R5)

**Files:**
- Modify: `train_sb3.py:34-44` (level tables), `train_sb3.py:56-89` (`CurriculumCallback`), `train_sb3.py:164-167` (`--no-curriculum` branch)
- Modify: `README.md` (Training section, Key parameters section)
- Test: `tests/test_curriculum.py`

**Interfaces:**
- Consumes: `BipedEnv.set_difficulty(push_max, terrain, speed)` (Task 1).
- Produces: level tuples `(push_max N*s, terrain, speed m/s)`. Exact tables:

```python
LEVELS_FIELD = [
    (0.0, 0.00, 0.0),    # stand still
    (0.0, 0.00, 0.3),
    (0.0, 0.00, 0.6),
    (0.0, 0.00, 1.0),
    (10.0, 0.00, 1.0),
    (25.0, 0.02, 1.0),
    (40.0, 0.04, 1.0),
    (60.0, 0.06, 1.0),
    (80.0, 0.08, 1.0),
]
LEVELS_COURSE = [(0.0, lv, 1.0) for lv in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)]  # resumed from a walking policy
```

- `CurriculumCallback._apply` unpacks three values, calls `env_method("set_difficulty", push, terrain, speed)`, prints `speed={speed} m/s`, and `_on_rollout_end` also records `self.logger.record("curriculum/speed", self.levels[self.level][2])`. The `--no-curriculum` branch unpacks and passes three values the same way.

- [ ] **Step 1: Write the failing tests in `tests/test_curriculum.py`**

```python
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_sb3 import LEVELS_FIELD, LEVELS_COURSE, CurriculumCallback

class FakeVecEnv:
    def __init__(self): self.calls = []
    def env_method(self, name, *args): self.calls.append((name, *args))

def test_levels_are_triples():
    assert all(len(l) == 3 for l in LEVELS_FIELD + LEVELS_COURSE)

def test_field_starts_standing_and_speed_never_drops():
    speeds = [l[2] for l in LEVELS_FIELD]
    assert speeds[0] == 0.0 and speeds == sorted(speeds) and speeds[-1] == 1.0

def test_apply_passes_speed():
    cb = CurriculumCallback(LEVELS_FIELD, start_level=1, verbose=0)
    cb.training_env = FakeVecEnv()
    cb._apply()
    assert cb.training_env.calls == [("set_difficulty", 0.0, 0.0, 0.3)]
```

- [ ] **Step 2: Run** `venv\Scripts\python -m pytest tests/test_curriculum.py -v`. Expected: all three fail (2-tuples; `_apply` passes two args).

- [ ] **Step 3: Implement** the Produces block.

- [ ] **Step 4: Run** `venv\Scripts\python -m pytest tests -v`. Expected: 13 passed. Then smoke-run `venv\Scripts\python train_sb3.py --name smoke --total-steps 20000 --n-envs 2 --dummy-vec`; expected: prints `[curriculum] level 0: push_max=0.0 N*s, terrain=0.0, speed=0.0 m/s` and `saved: runs\smoke`.

- [ ] **Step 5: Update `README.md`:** in Training, say the field curriculum first teaches standing (0 m/s), then walking at 0.3 / 0.6 / 1.0 m/s, then pushes and roughness; in Key parameters list the reward terms including `air_time` and `slip`; add `--scale air_time=0` to the ablation block.

- [ ] **Step 6: Commit** `git add tests/test_curriculum.py` then `git commit -am "feat: stand-then-walk speed curriculum"`

---

### Task 4: Decision experiment (R6)

No code. Two runs, ~1 h each at the observed ~1340 fps; they can run in parallel (12 CPU threads available) with `--n-envs 6` each.

- [ ] **Step 1: Run A (realistic gains):** `venv\Scripts\python train_sb3.py --name stand_a --total-steps 5000000 --n-envs 6 --save-every 250000`
- [ ] **Step 2: Run B (stiffer gains):** `venv\Scripts\python train_sb3.py --name stand_b --total-steps 5000000 --n-envs 6 --save-every 250000 --kp-scale 3`
- [ ] **Step 3: Watch** with `tensorboard --logdir runs` and `venv\Scripts\python watch.py --run runs/stand_a`.
- [ ] **Step 4: Judge against these criteria** (read `rollout/ep_len_mean`, `curriculum/level`, `curriculum/speed`):

| Checkpoint | Pass |
|---|---|
| 1M steps | `ep_len_mean` > 300 (6 s) |
| 2M steps | `curriculum/level` ≥ 1 (standing gate cleared) |
| 5M steps | `curriculum/speed` = 1.0 (level ≥ 3) |

- [ ] **Step 5: Decide.**
  - A passes → reward fix works; continue A to 20M for push/terrain levels.
  - A fails, B passes → control authority is the bottleneck: keep B's results, document `--kp-scale 3` in the report, try `--kp-scale 2` next.
  - Both fail at the 1M row → standing itself is not learned: inspect `info["slip"]`/`info["air_time"]` and the per-term contributions before changing more weights; do not start Stage 2.
  - Both pass the 1M row but stall at a speed level → tune that level's threshold or add an intermediate speed (e.g. 0.45) to `LEVELS_FIELD`.
- [ ] **Step 6:** Record the outcome (which row each run reached, at what step count) as one bullet under Notes in `README.md`; `git commit -am "docs: stand-then-walk experiment results"`.

---

## Out of scope (separate plans)

- Stage 2 port to MJX / MuJoCo Playground on Kaggle GPU — only after Task 4 passes, since every reward iteration on CPU costs ~1 h per 5M steps.
- Torque and joint-limit penalties, phase-based gait reward — add only if Task 4 shows unnatural gaits after walking is learned.
- Per-term reward logging to TensorBoard — add if Task 4 Step 5 needs it.
