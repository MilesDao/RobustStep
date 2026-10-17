# Unitree G1 walking to a goal: PPO in MuJoCo

Stage 1 of the roadmap: develop and debug on CPU. Stage 2 (MJX/Playground on Kaggle) comes after the reward is stable.

## Robot model

Uses the Unitree G1 (29 joints, 33.3 kg) from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie/tree/main/unitree_g1), the `g1_mjx.xml` version (simple colliders, realistic PD gains, already transferred to the real robot in MuJoCo Playground). The original file and `LICENSE` (BSD-3-Clause, Unitree Robotics) are in `g1/`. `g1/scene_g1.xml` is a custom scene: heightfield terrain, a goal sphere, and the collision pairs taken from `scene_mjx.xml` with the `floor` plane replaced by `terrain`.

If you use the MJX model in a paper, Menagerie suggests citing:
Zakka et al., *MuJoCo Playground*, arXiv:2502.08844.

The policy controls the 12 leg joints. The waist and both arms are held at the keyframe `home` pose.

## Installation

```
pip install -r requirements.txt
python check_env.py
```

`check_env.py` prints: the Gymnasium API, both feet's contacts, a standing test with simple ankle feedback, and whether random actions cause NaNs. The line "action = 0 falls at t=1.2s" is normal (see Notes).

## Training

```
python train_sb3.py --name run1 --total-steps 20000000 --n-envs 8
tensorboard --logdir runs/run1/tb
```

- The curriculum raises the level automatically when the no-fall rate over the last 50 episodes reaches 90%. The field curriculum first teaches standing (0 m/s), then walking at 0.3 / 0.6 / 1.0 m/s, then pushes and roughness. The levels are in `LEVELS_FIELD` / `LEVELS_COURSE` in `train_sb3.py` (max push impulse in N·s, roughness in m, speed in m/s).
- Checkpoints, `vecnormalize.pkl` and `env_kwargs.json` are saved in `runs/<name>/`. Evaluation and video rendering read these files back to use the right configuration.
- Watch the `fps` column in the log to estimate training time on your machine.
- `--kp-scale` multiplies the PD stiffness of every joint. 1.0 is the realistic gain. Larger values (e.g. 3) give the robot more control authority, so standing may be easier to learn, but it is less realistic. If you use it, state it in the report.

Ablations and baselines:

```
python train_sb3.py --name no_smooth --scale smooth=0
python train_sb3.py --name no_energy --scale action=0
python train_sb3.py --name no_air_time --scale air_time=0
python train_sb3.py --name flat_only --no-curriculum --level 0
```

Reward term names are in `DEFAULT_REWARD_SCALES` in `biped_env.py`.

## Evaluation

```
python evaluate.py --model runs/run1/latest.zip --stats runs/run1/vecnormalize.pkl --label full --out eval/full
```

Produces `push.csv`, `terrain.csv`, `summary.json`, `curves.png`: survival rate vs. impulse and the impulse at 50% survival, goal-reach rate, fall rate, time to goal, cost of transport (absolute mechanical energy divided by m·g·distance), action jerk.

## Recording video

```
python render_video.py --model runs/run1/latest.zip --stats runs/run1/vecnormalize.pkl --out demo.mp4 --push 40 --roughness 0.04
```

Linux and Kaggle need `MUJOCO_GL=egl`. The red arrow is the push force. The script uses the internal attribute `renderer._mjr_context` to upload new terrain to the GPU, so it may need adjusting if your MuJoCo version renames this attribute.

## Challenge corridor (scene `course`)

A 24 m x 6 m corridor (0.04 m grid cells); the robot starts at one end and the goal is a red sphere at the other. Each challenge is heightfield terrain generated in `terrains.py`. The difficulty `level` is in [0, 1] and maps to the physical parameters below (these are starting points; it has not been verified that the G1 can clear the maximum difficulty):

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

Preview the terrain (no policy needed): `MUJOCO_GL=egl python preview_terrains.py --out previews` (on Windows, drop the `MUJOCO_GL=egl` part).

Train on the corridor, continuing from a policy that already walks on flat ground:

```
python train_sb3.py --name slopes --scene course --challenge slope --resume runs/run1/latest.zip
```

For stairs, gaps, stepping stones, hurdles and beams, the robot needs to see the terrain, so enable `--height-scan` (adds 35 terrain-height observations, raising the observation size from 50 to 85, so you must train from scratch or from a policy that also used `--height-scan`).

Evaluate per challenge and per difficulty level:

```
python evaluate_challenges.py --model runs/slopes/latest.zip --stats runs/slopes/vecnormalize.pkl --kinds slope stairs --label slopes --out eval/slopes
```

Results are `challenges.csv`, `summary.json`, `challenges.png`, and the difficulty level (with physical parameters) at which the goal-reach rate drops to 50%. Add `--payload 5` (kg placed on the torso) or `--friction 0.3` to test load and slippery ground. Push force is still measured with `evaluate.py` as before.

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
- `g1/g1_mjx.xml` sets solver `iterations=5` to run fast on MJX. If foot contact looks unstable on CPU, increase this value.
