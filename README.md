# Unitree G1 walking to a goal: PPO in MuJoCo

Stage 1 of the roadmap: develop and debug on CPU. Stage 2 (MJX/Playground on Kaggle) comes after the reward is stable.

## Robot model

Uses the Unitree G1 (29 joints, 33.3 kg) from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1), the `g1_mjx.xml` version (simple colliders, realistic PD gains, already transferred to the real robot in MuJoCo Playground). The original file and `LICENSE` (BSD-3-Clause, Unitree Robotics) are in `assets/g1/`. `assets/g1/scene_g1.xml` is a custom scene: heightfield terrain, a goal sphere, and the collision pairs taken from `scene_mjx.xml` with the `floor` plane replaced by `terrain`.

If you use the MJX model in a paper, Menagerie suggests citing:
Zakka et al., *MuJoCo Playground*, arXiv:2502.08844.

The policy controls the 12 leg joints. The waist and both arms are held at the keyframe `home` pose.

## Project layout

```
biped_rl/        core package: env.py (Gymnasium env), terrains.py, progress_data.py
scripts/         everything you run: train.py, evaluate.py, evaluate_challenges.py,
                 render_video.py, progress_video.py, watch.py, preview_terrains.py, check_env.py
assets/g1/       G1 model and scenes (MuJoCo Menagerie, BSD-3-Clause)
tests/           pytest suite (`pytest` from the project root)
runs/<name>/     training runs: checkpoints, vecnormalize, env_kwargs.json, monitor CSVs, tb/, train.log
runs/archive/    old runs kept for reference
outputs/         videos, charts, evaluation results, terrain previews
docs/            design notes and implementation plans
```

Run every command from the project root.

## Installation

```
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py
```

`scripts/check_env.py` prints: the Gymnasium API, both feet's contacts, a standing test with simple ankle feedback, and whether random actions cause NaNs. The line "action = 0 falls at t=1.2s" is normal (see Notes).

## Training

```
python scripts/train.py --name run1 --total-steps 20000000 --n-envs 8
tensorboard --logdir runs/run1/tb
```

- The curriculum raises the level automatically when the no-fall rate over the last 50 episodes reaches 90%. The field curriculum first teaches standing (0 m/s), then walking at 0.3 / 0.6 / 1.0 m/s, then pushes and roughness. The levels are in `LEVELS_FIELD` / `LEVELS_COURSE` in `scripts/train.py` (max push impulse in N·s, roughness in m, speed in m/s). Level indices 0-2 are the standing / slow-walking stages, level 3 is flat walking at 1.0 m/s, and pushes and roughness start at level 4. At speed levels above 0 the gate also requires the robot to cover at least 70% of the commanded distance, so standing still does not advance.
- Checkpoints, `vecnormalize.pkl` and `env_kwargs.json` are saved in `runs/<name>/`. Evaluation and video rendering read these files back to use the right configuration.
- Watch the `fps` column in the log to estimate training time on your machine.
- `--kp-scale` multiplies the PD stiffness of every joint. 1.0 is the realistic gain. Larger values (e.g. 3) give the robot more control authority, so standing may be easier to learn, but it is less realistic. If you use it, state it in the report.

Ablations and baselines:

```
python scripts/train.py --name no_smooth --scale smooth=0
python scripts/train.py --name no_energy --scale action=0
python scripts/train.py --name no_air_time --scale air_time=0
python scripts/train.py --name flat_only --no-curriculum --level 3
```

Reward term names are in `DEFAULT_REWARD_SCALES` in `biped_rl/env.py`.

### Training on Kaggle

`kaggle/train_kaggle.ipynb` clones this repo from GitHub and trains on a Kaggle machine (more RAM, runs unattended). Push the branch first, import the notebook, turn Internet on, edit `REPO_URL`/`BRANCH`/`RUN`/`START_LEVEL`/`STEPS` in the first cell (private repo: add a `GITHUB_TOKEN` Kaggle secret), then *Save & Run All*. Checkpoints are gitignored, so to continue a run, upload its newest `ckpt_<N>_steps.zip` + `ckpt_vecnormalize_<N>_steps.pkl` as a small Kaggle Dataset and attach it; without one it trains from scratch. It stops itself at `MAX_HOURS` (11; Kaggle kills at 12 h) and still saves. Download `<name>_out.zip` from Output and unzip it into `runs/`. The physics runs on the CPU, so a GPU accelerator does not speed this up; GPU training needs the MJX port (stage 2).

## Evaluation

```
python scripts/evaluate.py --model runs/run1/latest.zip --stats runs/run1/vecnormalize.pkl --label full --out outputs/eval/full
```

Produces `push.csv`, `terrain.csv`, `summary.json`, `curves.png`: survival rate vs. impulse and the impulse at 50% survival, goal-reach rate, fall rate, time to goal, cost of transport (absolute mechanical energy divided by m·g·distance), action jerk.

## Recording video

```
python scripts/render_video.py --model runs/run1/latest.zip --stats runs/run1/vecnormalize.pkl --out outputs/videos/demo.mp4 --push 40 --roughness 0.04
```

Linux and Kaggle need `MUJOCO_GL=egl`. The red arrow is the push force. The script uses the internal attribute `renderer._mjr_context` to upload new terrain to the GPU, so it may need adjusting if your MuJoCo version renames this attribute.

## Learning-progress video

`scripts/progress_video.py` renders one fixed-seed clip per checkpoint and joins them into one MP4: title card, "Part 1 — Learning to walk" (field run), "Part 2 — Challenges" (course run), and a card at each curriculum level-up. A chart of survival time vs attempt is saved next to the video as `<name>_chart.png`. Each clip shows the attempt number, simulated practice time, training steps, the curriculum stage, and a live clock with distance walked; a fall freezes the frame with "FELL at … s".

```
python scripts/train.py --name walk_scan --height-scan --total-steps 10000000 --n-envs 10 --save-every 200000
python scripts/train.py --name course_mixed --scene course --challenge mixed --height-scan --resume runs/walk_scan/latest.zip --total-steps 10000000 --n-envs 10 --save-every 200000
python scripts/progress_video.py --runs runs/walk_scan runs/course_mixed --challenges stairs slope gaps stones --every 2 --out outputs/videos/g1_learning_progress.mp4
```

Attempt counts are exact for runs trained after Monitor CSV logging was added (`runs/<name>/monitor_*.monitor.csv`); older runs show an estimate from TensorBoard, marked `≈`. Use `--width 1080 --height 1920` for a vertical cut, `--every`/`--clip-seconds` to change the length.

## Challenge corridor (scene `course`)

A 24 m x 6 m corridor (0.04 m grid cells); the robot starts at one end and the goal is a red sphere at the other. Each challenge is heightfield terrain generated in `biped_rl/terrains.py`. The difficulty `level` is in [0, 1] and maps to the physical parameters below (these are starting points; it has not been verified that the G1 can clear the maximum difficulty):

| Challenge | Level 0 | Level 1 |
|---|---|---|
| `slope` (up then down) | 3° | 15° |
| `stairs` (steps up then down, 30 cm tread) | 4 cm step | 15 cm step |
| `gaps` (3 gaps across the corridor, 40 cm deep) | 10 cm gap | 30 cm gap |
| `stones` (stepping stones, 1.2 m wide path, pits around) | 10 cm gap, 50 cm stones | 30 cm gap, 35 cm stones |
| `hurdles` (3 crossbars) | 5 cm high | 20 cm high |
| `beam` (3 m balance beam, pits around) | 50 cm wide | 20 cm wide |
| `rough` (rough ground) | 2 cm amplitude | 10 cm amplitude |
| `mixed` | random mix of the above | |

Preview the terrain (no policy needed): `MUJOCO_GL=egl python scripts/preview_terrains.py --out outputs/previews` (on Windows, drop the `MUJOCO_GL=egl` part).

Train on the corridor, continuing from a policy that already walks on flat ground:

```
python scripts/train.py --name slopes --scene course --challenge slope --resume runs/run1/latest.zip
```

For stairs, gaps, stepping stones, hurdles and beams, the robot needs to see the terrain, so enable `--height-scan` (adds 35 terrain-height observations, raising the observation size from 50 to 85, so you must train from scratch or from a policy that also used `--height-scan`).

Evaluate per challenge and per difficulty level:

```
python scripts/evaluate_challenges.py --model runs/slopes/latest.zip --stats runs/slopes/vecnormalize.pkl --kinds slope stairs --label slopes --out outputs/eval/slopes
```

Results are `challenges.csv`, `summary.json`, `challenges.png`, and the difficulty level (with physical parameters) at which the goal-reach rate drops to 50%. Add `--payload 5` (kg placed on the torso) or `--friction 0.3` to test load and slippery ground. Push force is still measured with `scripts/evaluate.py` as before.

The robot counts as fallen when it tilts more than 60°, the hip is low relative to the feet, the pelvis is 30 cm below the ground beneath it (fell into a pit), or it leaves the corridor.

## Key parameters

- Policy at 50 Hz (timestep 0.004 s, frame_skip 5). The action is a joint-angle offset around the `home` pose, driven by MuJoCo's PD controller.
- 50-dimensional observation (85 with `--height-scan`): gravity direction, angular and linear velocity of the pelvis, angles and velocities of the 12 leg joints, previous action, direction and distance to the goal, contacts of both feet.
- Reward terms (`DEFAULT_REWARD_SCALES`): `vel`, `heading`, `upright`, `height`, `alive`, `action`, `smooth`, `ang_vel`, `vz`, `reach`, `fall`, `air_time`, `slip`.
- Domain randomization when `randomize=True`: mass ±15%, friction ±30%.
- The episode ends when the pelvis tilts more than 60° (up_z < 0.5) or is lower than 0.45 m relative to the feet.

## Notes

- With realistic gains (ankle kp 20), action = 0 cannot stand by itself: the G1 falls forward after about 1.2 s. This is correct physical behavior (inverted pendulum); the policy must learn to balance. Control authority was verified: with `kp_scale=4` and a simple ankle feedback controller, the G1 stands for 8 seconds.
- The escalator is not implemented. It would need the steps to be moving bodies (slide joints with velocity actuators) plus their own collision pairs, unlike the static heightfield terrain used here.
- No policy has been fully trained yet. The 40k-step model was only a pipeline smoke test.
- The G1 is far more complex than a simple biped character, so it needs many more training steps and may need reward tuning (e.g. a gait-rhythm reward if the policy learns to shuffle its feet). Reward weights and curriculum thresholds are only starting values.
- `assets/g1/g1_mjx.xml` sets solver `iterations=5` to run fast on MJX. If foot contact looks unstable on CPU, increase this value.
