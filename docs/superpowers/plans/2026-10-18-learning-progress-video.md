# Learning-Progress Video Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one MP4 that shows the G1 learning to walk from scratch on flat ground, then learning to cross the course challenges, with attempt count, practice time and curriculum milestones overlaid.

**Architecture:** Training already saves `ckpt_<steps>_steps.zip` + matching VecNormalize every `--save-every` steps. A small data module turns a run folder into an ordered list of checkpoints annotated with attempts, curriculum level and speed (from Monitor CSVs when present, TensorBoard otherwise). The existing offscreen rollout in `render_video.py` is extracted into a reusable function. A new `progress_video.py` renders one fixed-seed clip per checkpoint, draws overlays with PIL, inserts milestone cards and an ending chart, and writes one MP4. Two new training runs provide the material for the course part.

**Tech Stack:** Python, MuJoCo ≥3.2 (offscreen `mujoco.Renderer`), Stable-Baselines3 2.9, imageio + imageio-ffmpeg, Pillow 13 (already installed with matplotlib), tensorboard 2.21 (`EventAccumulator`), matplotlib, pytest.

**Spec:** No separate file; the "Video spec" section below is the spec (agreed in chat on 2026-10-17/18).

## Video spec

Per-clip overlay (top-left, white text on a semi-transparent dark box):
1. `Attempt #12,345` — cumulative episodes before this checkpoint (prefix `≈` when estimated).
2. `Practice: 3.2 h simulated` — `steps × 0.02 s`.
3. `Training steps: 1.25M`.
4. Stage line, field run: `Stage: Stand still` / `Walk 0.3 m/s` / `Walk 0.6 m/s` / `Walk 1.0 m/s` / `Pushes up to 25 N·s, rough 2 cm`; course run: `Stairs: step 9.5 cm` (from `terrains.describe`).
5. Live clip clock `t = 4.3 s`, plus distance `2.1 m`; the clock turns red and a `FELL at 1.6 s` banner shows when the episode terminates by falling, with the last frame held 0.5 s.

Video structure: title card (3 s) → Part 1 "Learning to walk" (one clip per selected field checkpoint) → Part 2 "Challenges" (one clip per selected course checkpoint, rotating through the chosen challenge kinds at the difficulty the curriculum had reached) → milestone cards (2 s) inserted wherever the curriculum level increased since the previous clip → ending chart (4 s): survival time of each clip vs attempt number, the current clip's dot highlighted as the chart builds up.

Defaults (user did not specify): 16:9, 1280×720, 25 fps, English text. Vertical 1080×1920 must work via `--width/--height` without code changes.

Every clip of the same part uses the same seed and start, so differences come from learning, not luck.

## Global Constraints

- **No git operations.** Never run `git add`, `git commit`, `git push` (user rule in `~/.claude/CLAUDE.md`). Leave all changes uncommitted and list changed files in the report. This replaces every "commit" step.
- No new dependencies. Pillow, tensorboard, imageio(-ffmpeg), matplotlib are already installed in `venv/`.
- Observation dimension unchanged (50, or 85 with `--height-scan`); `biped_env.py` is not modified by this plan.
- Policy step `dt = 0.02 s`; practice time = steps × 0.02 s.
- `render_video.py` command-line behavior stays identical after the refactor.
- Comments/strings in English, short inline `#` comments, one-line docstrings, matching existing files.
- Use `venv/Scripts/python` on Windows. If offscreen GL fails, `MUJOCO_GL=glfw` (Windows default) / `egl` (Linux).

## Review Focus

1. Checkpoint step counts are not round (`ckpt_249996_steps.zip` with 6 envs): sorting must be numeric, labels must round (`0.25M`) → `test_list_checkpoints_sorted_numerically` (Task 2).
2. A checkpoint zip without its VecNormalize pkl (written a moment later, or deleted) must be skipped, not crash → `test_list_checkpoints_skips_missing_stats` (Task 2).
3. Runs started before Task 1 (`stand_a`, `stand_b`) have no Monitor CSV: attempts must fall back to the TensorBoard estimate and be flagged inexact → `test_attempts_falls_back_to_tensorboard` (Task 2).
4. A resumed run (course run D from C) restarts its own Monitor CSV and step counter: Part 2 attempt numbers must continue from Part 1's last value, not restart at 0 → `test_offsets_chain_runs` (Task 4).
5. A clip where the robot never falls must not show a FELL banner and must not hold a frame; a clip that falls at step 1 must still yield at least one frame → `test_rollout_frames_fall_and_no_fall` (Task 3).

---

## File Structure

- Modify `train_sb3.py:52-58` (`make_env`), `:143` (call site): Monitor writes CSVs into the run folder (Task 1).
- Create `progress_data.py`: run folder → checkpoint list, attempts, TensorBoard scalars. Pure, no rendering (Task 2).
- Modify `render_video.py`: extract `rollout_frames(...)`; `main()` uses it (Task 3).
- Create `progress_video.py`: overlay drawing, cards, chart, CLI that assembles the MP4 (Task 4).
- Modify `README.md`: a "Learning-progress video" section (Task 4).
- Create `tests/test_train_monitor.py`, `tests/test_progress_data.py`, `tests/test_render.py`, `tests/test_progress_video.py`.

---

### Task 1: Exact attempt logging

**Files:**
- Modify: `train_sb3.py:52-58`, `train_sb3.py:143`
- Test: `tests/test_train_monitor.py`

**Interfaces:**
- Produces: `make_env(rank: int, seed: int, env_kwargs: dict, log_dir: str | None = None)`. With `log_dir`, Monitor is created with `filename=os.path.join(log_dir, f"monitor_{rank}")`, so SB3 writes `<run>/monitor_<rank>.monitor.csv` (columns `r,l,t,fell,n_reached,path`, one row per finished episode). `main()` passes `log_dir=out_dir`.

- [ ] **Step 1: Write the failing test** (same `sys.path.insert` header as `tests/test_reward.py`)

```python
import glob, numpy as np
from train_sb3 import make_env

def test_monitor_csv_written(tmp_path):
    env = make_env(0, 0, {"randomize": False}, log_dir=str(tmp_path))()
    env.reset(seed=0)
    for _ in range(2000):                       # zero action falls in ~1-2 s
        *_, term, trunc, _ = env.step(np.zeros(12))
        if term or trunc:
            break
    env.close()
    files = glob.glob(str(tmp_path / "monitor_0*.csv"))
    assert len(files) == 1
    lines = open(files[0]).read().splitlines()
    assert lines[1] == "r,l,t,fell,n_reached,path" and len(lines) == 3
```

- [ ] **Step 2:** `venv/Scripts/python -m pytest tests/test_train_monitor.py -v` → FAIL (`TypeError: unexpected keyword 'log_dir'`).
- [ ] **Step 3:** Implement the Produces block.
- [ ] **Step 4:** Same command → PASS; full suite `venv/Scripts/python -m pytest tests -q` → all pass. Smoke: `venv/Scripts/python train_sb3.py --name smoke --total-steps 20000 --n-envs 2 --dummy-vec`, then `ls runs/smoke/*.monitor.csv` shows 2 files; delete `runs/smoke`.
- [ ] **Step 5:** No commit (Global Constraints). Report changed files.

---

### Task 2: Run-folder data module

**Files:**
- Create: `progress_data.py`
- Test: `tests/test_progress_data.py`

**Interfaces:**
- Consumes: Monitor CSV format from Task 1; TB tags `rollout/ep_len_mean`, `curriculum/level`, `curriculum/speed` written by `train_sb3.py`; `train_sb3.py` PPO `n_steps = 1024` per env.
- Produces (module `progress_data`):
  - `@dataclass Checkpoint: steps: int; model: str; stats: str`
  - `list_checkpoints(run_dir: str) -> list[Checkpoint]` — parses `ckpt_<N>_steps.zip`, pairs with `ckpt_vecnormalize_<N>_steps.pkl`, skips zips without a pkl, sorted by `steps` ascending (numeric).
  - `attempts_at(run_dir: str, steps: int) -> tuple[int, bool]` — `(episodes finished before `steps`, exact)`. Exact path: for each `monitor_*.monitor.csv`, per-env cumulative `l`; count rows whose cumulative `l <= steps / n_csv_files`; sum over files; `exact=True`. Fallback when no CSV: TB estimate `sum over logged points p with p.step <= steps of (p.step - prev_step) / p.value` using `rollout/ep_len_mean`; `exact=False`. No CSV and no TB → `(0, False)`.
  - `scalar_at(run_dir: str, tag: str, steps: int) -> float | None` — value of the last TB point with `step <= steps` (TB event files under `<run_dir>/tb/**`); `None` if the tag is absent.
  - `format_steps(steps: int) -> str` — `249996 -> "0.25M"`, `1250000 -> "1.25M"`, `5000000 -> "5M"` (2 decimals, trailing zeros and dot stripped).

- [ ] **Step 1: Write failing tests** (TB fixtures written with `torch.utils.tensorboard.SummaryWriter(str(tmp_path / "tb" / "ppo_1"))`, `add_scalar(tag, value, step)`, `close()`):

```python
def test_list_checkpoints_sorted_numerically(tmp_path):
    for n in (999984, 249996, 1249980):
        (tmp_path / f"ckpt_{n}_steps.zip").write_bytes(b"")
        (tmp_path / f"ckpt_vecnormalize_{n}_steps.pkl").write_bytes(b"")
    assert [c.steps for c in list_checkpoints(str(tmp_path))] == [249996, 999984, 1249980]

def test_list_checkpoints_skips_missing_stats(tmp_path):
    (tmp_path / "ckpt_100_steps.zip").write_bytes(b"")
    assert list_checkpoints(str(tmp_path)) == []

def test_attempts_exact_from_monitor(tmp_path):
    for rank, ls in ((0, [10, 20, 30]), (1, [60])):
        rows = "\n".join(f"0,{l},0,1,0,0" for l in ls)
        (tmp_path / f"monitor_{rank}.monitor.csv").write_text('#{"t_start": 0}\nr,l,t,fell,n_reached,path\n' + rows + "\n")
    # 2 files, steps=60 -> 30 per env: env0 cumulative 10,30,60 -> 2 ; env1 cumulative 60 -> 0
    assert attempts_at(str(tmp_path), 60) == (2, True)

def test_attempts_falls_back_to_tensorboard(tmp_path):
    w = SummaryWriter(str(tmp_path / "tb" / "ppo_1"))
    w.add_scalar("rollout/ep_len_mean", 50.0, 1000); w.add_scalar("rollout/ep_len_mean", 100.0, 3000); w.close()
    assert attempts_at(str(tmp_path), 3000) == (40, False)    # 1000/50 + 2000/100

def test_scalar_at_last_point_before(tmp_path):
    w = SummaryWriter(str(tmp_path / "tb" / "ppo_1"))
    w.add_scalar("curriculum/level", 0, 100); w.add_scalar("curriculum/level", 2, 500); w.close()
    assert scalar_at(str(tmp_path), "curriculum/level", 499) == 0
    assert scalar_at(str(tmp_path), "curriculum/level", 500) == 2
    assert scalar_at(str(tmp_path), "curriculum/speed", 500) is None

def test_format_steps():
    assert [format_steps(s) for s in (249996, 1250000, 5000000)] == ["0.25M", "1.25M", "5M"]
```

- [ ] **Step 2:** `venv/Scripts/python -m pytest tests/test_progress_data.py -v` → FAIL (`ModuleNotFoundError: progress_data`).
- [ ] **Step 3:** Implement the Produces block. Use `tensorboard.backend.event_processing.event_accumulator.EventAccumulator` per event file (glob `tb/**/events.out.tfevents.*`, recursive), merge points of all files, sort by step. Read CSVs with the `csv` module, skipping the leading `#` JSON line.
- [ ] **Step 4:** Same command → 6 passed. Manual check on the live run: `venv/Scripts/python -c "import progress_data as p; c=p.list_checkpoints('runs/stand_a'); print([(x.steps, p.attempts_at('runs/stand_a', x.steps), p.scalar_at('runs/stand_a','curriculum/level',x.steps)) for x in c])"` prints increasing attempts flagged `False` (stand_a predates Task 1).
- [ ] **Step 5:** No commit. Report changed files.

---

### Task 3: Reusable rollout renderer

**Files:**
- Modify: `render_video.py` (move the loop in `main()` lines ~59-80 into a function; `main()` calls it)
- Test: `tests/test_render.py`

**Interfaces:**
- Produces in `render_video.py`:
  - `make_renderer(env: BipedEnv, width: int, height: int) -> tuple[mujoco.Renderer, mujoco.MjvCamera]` — creates the renderer, uploads the hfield (`mjr_uploadHField(..., renderer._mjr_context, 0)`), tracking camera on `env.torso_id`, `distance, azimuth, elevation = 4.0, 120, -12`. Must be called after `env.reset()` (terrain is generated at reset).
  - `rollout_frames(env, act, obs, renderer, cam, seconds: float, fps: int) -> tuple[list[np.ndarray], dict]` — steps the env from the given `obs`, renders every `round(1/(fps*env.dt))` steps (push arrow drawn as today via `add_arrow`), stops on `terminated`/`truncated`/time. Returns frames (always ≥1: render once before the first step) and `{"fell": bool, "t": float, "path": float}` where `t` is sim time at the end and `fell = terminated and not info["success"]`. Each frame is paired with its sim time: frames is a list of `(t, image)` tuples so overlays can show the clock.
  - `main()` keeps the same CLI and output; it saves `[img for _, img in frames]`.

- [ ] **Step 1: Write the failing test**

```python
from biped_env import BipedEnv
from render_video import make_renderer, rollout_frames

def test_rollout_frames_fall_and_no_fall():
    env = BipedEnv(randomize=False)
    obs, _ = env.reset(seed=0)
    r, cam = make_renderer(env, 64, 48)
    frames, info = rollout_frames(env, lambda o: np.zeros(12), obs, r, cam, seconds=5.0, fps=10)
    assert info["fell"] and info["t"] < 5.0                  # zero action falls (~1.2 s)
    assert len(frames) >= 1 and frames[0][1].shape == (48, 64, 3)
    obs, _ = env.reset(seed=0)
    frames, info = rollout_frames(env, lambda o: np.zeros(12), obs, r, cam, seconds=0.2, fps=10)
    assert not info["fell"] and len(frames) == 3             # t=0 + 2 frames in 0.2 s
```

- [ ] **Step 2:** `venv/Scripts/python -m pytest tests/test_render.py -v` → FAIL (ImportError).
- [ ] **Step 3:** Implement the Produces block.
- [ ] **Step 4:** Test passes; full suite passes; `venv/Scripts/python render_video.py --model runs/stand_a/ckpt_999984_steps.zip --stats runs/stand_a/ckpt_vecnormalize_999984_steps.pkl --out scratch_demo.mp4 --seconds 4` writes a playable MP4 (delete it after).
- [ ] **Step 5:** No commit. Report changed files.

---

### Task 4: `progress_video.py`

**Files:**
- Create: `progress_video.py`
- Modify: `README.md` (new section "Learning-progress video": the two commands from Task 5 Step 3/4 and one line on what the overlay shows)
- Test: `tests/test_progress_video.py`

**Interfaces:**
- Consumes: `progress_data.list_checkpoints / attempts_at / scalar_at / format_steps` (Task 2); `render_video.make_renderer / rollout_frames` (Task 3); `evaluate.load_policy(model, stats)`, `evaluate.load_env_kwargs(stats)`, `evaluate.load_train_scene(stats)`; `train_sb3.LEVELS_FIELD`, `train_sb3.LEVELS_COURSE` (tuples `(push, terrain, speed)`); `terrains.describe(kind, level) -> dict`.
- Produces in `progress_video.py`:
  - `draw_overlay(img: np.ndarray, lines: list[str], banner: str | None = None, clock_red: bool = False) -> np.ndarray` — returns a new uint8 array of the same shape; text box top-left, banner centered in red when given. Font: `ImageFont.load_default(size=max(14, h // 30))` (Pillow ≥10.1 supports `size`).
  - `card(text: str, w: int, h: int, seconds: float, fps: int) -> list[np.ndarray]` — black frames with centered white text (multi-line via `\n`).
  - `chart_frames(attempts: list[int], survival: list[float], w: int, h: int, seconds: float, fps: int) -> list[np.ndarray]` — matplotlib (Agg) line+dots of survival time vs attempt; points appear one by one over `seconds`, newest dot highlighted.
  - `stage_text(scene: str, level: int | None, kind: str | None) -> str` — field: speed 0 → `"Stand still"`, push 0 & speed > 0 → `"Walk {speed} m/s"`, push > 0 → `"Pushes up to {push:g} N·s, rough {terrain*100:g} cm"`; course: `"{Kind}: " + ", ".join(f"{k} {v}" for k, v in terrains.describe(kind, LEVELS_COURSE[level][1]).items())`; `level is None` → `"Stage: unknown"`.
  - `chain_offsets(run_dirs: list[str], last_steps: list[int]) -> list[tuple[int, int]]` — per run `(attempt_offset, step_offset)`: cumulative attempts and steps of all previous runs, using `attempts_at(prev_run, prev_last_steps)[0]` and `prev_last_steps`.
  - CLI: `python progress_video.py --runs RUN [RUN ...] --out FILE.mp4 [--every K] [--clip-seconds 8] [--challenges stairs gaps ...] [--width 1280 --height 720 --fps 25] [--seed 0] [--title "G1 learns to walk"]`. Each run's scene comes from `load_train_scene`. `--every K` keeps every K-th checkpoint plus always the last one. Course clips cycle through `--challenges` (default `stairs slope gaps`) and render at `difficulty = LEVELS_COURSE[int(level)][1]` read via `scalar_at(run, "curriculum/level", ckpt.steps)` (0 if None); field clips use `opts = {"on_reach": "resample"}` plus roughness/push from that level's tuple. Overlay lines exactly as in the Video spec; attempts and steps include the offsets from `chain_offsets`; prefix `≈` when `exact` is False. Fell → `banner="FELL at {t:.1f} s"`, `clock_red=True`, last frame repeated `fps // 2` times. Part cards: `"Part 1 — Learning to walk"` before the first field clip, `"Part 2 — Challenges"` before the first course clip. Milestone card `"Attempt #{n}: {stage_text}"` when the curriculum level differs from the previous clip's. Ending `chart_frames` over all clips (survival = `info["t"]`). Write with `imageio.mimsave(out, frames, fps=fps)` (same gif fallback as `render_video.py`).

- [ ] **Step 1: Write failing tests**

```python
def test_draw_overlay_keeps_shape_and_changes_pixels():
    img = np.zeros((72, 128, 3), np.uint8)
    out = draw_overlay(img, ["Attempt #1"], banner="FELL at 1.6 s", clock_red=True)
    assert out.shape == img.shape and out.dtype == np.uint8 and out.any() and not img.any()

def test_card_and_chart_sizes():
    assert len(card("Hi", 128, 72, 1.0, 10)) == 10
    f = chart_frames([1, 10, 100], [1.2, 3.0, 20.0], 128, 72, 1.0, 10)
    assert len(f) == 10 and f[0].shape == (72, 128, 3)

def test_stage_text():
    assert stage_text("field", 0, None) == "Stand still"
    assert stage_text("field", 1, None) == "Walk 0.3 m/s"
    assert stage_text("field", 5, None) == "Pushes up to 25 N·s, rough 2 cm"
    assert stage_text("course", 0, "stairs").startswith("Stairs: step height (cm) 4.0")
    assert stage_text("field", None, None) == "Stage: unknown"

def test_offsets_chain_runs(monkeypatch):
    monkeypatch.setattr(progress_video, "attempts_at", lambda run, steps: ({"A": 500}[run], True))
    assert chain_offsets(["A", "B"], [1_000_000, 300_000]) == [(0, 0), (500, 1_000_000)]
```

- [ ] **Step 2:** `venv/Scripts/python -m pytest tests/test_progress_video.py -v` → FAIL (ModuleNotFoundError).
- [ ] **Step 3:** Implement the Produces block (`import progress_data` names into `progress_video` with `from progress_data import attempts_at, ...` so the monkeypatch target works).
- [ ] **Step 4:** Tests pass; full suite passes. End-to-end: `venv/Scripts/python progress_video.py --runs runs/stand_a --out scratch_progress.mp4 --clip-seconds 3 --width 640 --height 360` → MP4 exists, opens, shows title card, clips with overlay, ending chart. Delete after checking.
- [ ] **Step 5:** README section. No commit. Report changed files.

---

### Task 5: Record the material and render the final video

No code. Compute-heavy; run in the background.

Why two new runs: course challenges (stairs, gaps, stones, hurdles, beam) need `--height-scan`, which makes the observation 85-dim, so `stand_a` (50-dim) cannot be resumed on the course. Run C learns to walk with the height scan on the field (same stand-then-walk curriculum); run D resumes C on the course.

- [ ] **Step 1: Gate.** Read `stand_a`/`stand_b` results (TensorBoard `rollout/ep_len_mean`, `curriculum/level`, `curriculum/track_recent`). Start Step 2 only if one of them reached `curriculum/speed = 1.0`; use that run's `--kp-scale` (1 for A, 3 for B) for C and D. If neither did, stop: the reward needs work before a video is worth recording.
- [ ] **Step 2: Run C (field, height scan, exact attempts, denser checkpoints):**
  `venv/Scripts/python train_sb3.py --name walk_scan --height-scan --total-steps 10000000 --n-envs 10 --save-every 200000 [--kp-scale 3]`
  Pass: `curriculum/level` ≥ 3 (walking 1.0 m/s on flat).
- [ ] **Step 3: Run D (course, resumed from C):**
  `venv/Scripts/python train_sb3.py --name course_mixed --scene course --challenge mixed --height-scan --resume runs/walk_scan/latest.zip --total-steps 10000000 --n-envs 10 --save-every 200000 [--kp-scale 3]`
  Watch `curriculum/level` (course difficulty 0 → 1.0 in steps of 0.2). If it stalls with `survival_recent` ≥ 0.9 but `track_recent` low, the 0.7 tracking ratio is too strict for obstacles: lower `CurriculumCallback(track_ratio=...)` for the course run and resume.
- [ ] **Step 4: Render:**
  `venv/Scripts/python progress_video.py --runs runs/walk_scan runs/course_mixed --challenges stairs slope gaps stones --every 2 --out g1_learning_progress.mp4`
  Expected length: (~50 + ~50 checkpoints)/2 × ~8 s + cards ≈ 7 min; raise `--every` or lower `--clip-seconds` for a ~2-3 min cut. Vertical cut: add `--width 1080 --height 1920`.
- [ ] **Step 5:** Note in README Notes which runs produced the video and their final curriculum levels. No commit.

---

## Out of scope

- Splitting several policies into one frame (side-by-side grid) — add after the single-track video exists if wanted.
- Music, voice-over, YouTube upload — done by the user in a video editor.
- Re-rendering `stand_a` with exact attempt counts — impossible without its Monitor CSVs; it can still be used with `≈` estimates.
