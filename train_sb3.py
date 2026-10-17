"""PPO training (Stable-Baselines3, CPU) for BipedEnv, with curriculum.

Two modes:
  --scene field   (default): walk to the goal on random terrain, curriculum over push force and roughness.
  --scene course : challenge corridor (slope, stairs, gap, stepping stones, obstacle, beam), curriculum over difficulty.

Examples:
  python train_sb3.py --name run1 --total-steps 20000000 --n-envs 8
  python train_sb3.py --name no_smooth --scale smooth=0            # ablation of one reward term
  python train_sb3.py --name flat_only --no-curriculum --level 0   # baseline trained on flat ground only
  python train_sb3.py --name stiff --kp-scale 3                     # stiffer PD, easier to learn standing, less realistic
  python train_sb3.py --name stairs1 --scene course --challenge stairs --height-scan \\
         --resume runs/run1/latest.zip                              # keep fine-tuning on stairs
  python train_sb3.py --name run1 --resume runs/run1/latest.zip --start-level 2

Note: enabling --height-scan changes the observation dimension (50 -> 85), so you cannot resume from a policy
trained without the height sensor.
"""
import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")  # each environment process uses only 1 thread

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecNormalize

from biped_env import BipedEnv

# (push_max N*s, terrain, speed m/s). For scene field, terrain is roughness (m). For scene course, it is difficulty [0, 1].
# The field curriculum first teaches standing, then walking speeds, then pushes and roughness. Adjust based on actual results.
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


def make_env(rank, seed, env_kwargs):
    def _init():
        env = BipedEnv(**env_kwargs)
        env = Monitor(env, info_keywords=("fell", "n_reached"))
        env.reset(seed=seed + rank)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """Raise the level when the no-fall episode rate over the last 50 episodes reaches the threshold."""

    def __init__(self, levels, start_level=0, survive_threshold=0.9, min_episodes=50, verbose=1):
        super().__init__(verbose)
        self.levels = levels
        self.level = start_level
        self.survive_threshold = survive_threshold
        self.min_episodes = min_episodes

    def _apply(self):
        push, terrain, speed = self.levels[self.level]
        self.training_env.env_method("set_difficulty", push, terrain, speed)
        if self.verbose:
            print(f"[curriculum] level {self.level}: push_max={push} N*s, terrain={terrain}, speed={speed} m/s", flush=True)

    def _on_training_start(self):
        self._apply()

    def _on_step(self):
        return True

    def _on_rollout_end(self):
        buf = self.model.ep_info_buffer
        self.logger.record("curriculum/level", self.level)
        self.logger.record("curriculum/speed", self.levels[self.level][2])
        if len(buf) < self.min_episodes or self.level >= len(self.levels) - 1:
            return
        recent = list(buf)[-self.min_episodes:]
        survive = 1.0 - float(np.mean([e["fell"] for e in recent]))
        self.logger.record("curriculum/survival_recent", survive)
        if survive >= self.survive_threshold:
            self.level += 1
            buf.clear()
            self._apply()


def parse_scales(items):
    out = {}
    for it in items or []:
        k, v = it.split("=")
        out[k] = float(v)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", default="run1")
    p.add_argument("--total-steps", type=int, default=20_000_000)
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--scene", choices=["field", "course"], default="field")
    p.add_argument("--challenge", default="mixed",
                   help="scene course: rough slope stairs gaps stones hurdles beam mixed flat")
    p.add_argument("--height-scan", action="store_true", help="add 35 terrain-height observations around the robot")
    p.add_argument("--scan-noise", type=float, default=0.01, help="noise (m) of the height observation during training")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--level", type=int, default=0, help="fixed level when using --no-curriculum")
    p.add_argument("--start-level", type=int, default=0)
    p.add_argument("--scale", action="append", help="override reward weights, e.g. smooth=0")
    p.add_argument("--kp-scale", type=float, default=1.0,
                   help="multiplies the PD stiffness of every joint (1.0 = realistic Menagerie/Playground gain)")
    p.add_argument("--resume", default=None)
    p.add_argument("--save-every", type=int, default=1_000_000)
    p.add_argument("--dummy-vec", action="store_true", help="use DummyVecEnv (debug, no multiprocessing)")
    args = p.parse_args()

    torch.set_num_threads(4)
    out_dir = os.path.join("runs", args.name)
    os.makedirs(out_dir, exist_ok=True)

    env_kwargs = dict(reward_scales=parse_scales(args.scale), kp_scale=args.kp_scale, scene=args.scene,
                      challenge=args.challenge, height_scan=args.height_scan,
                      scan_noise=args.scan_noise if args.height_scan else 0.0)
    with open(os.path.join(out_dir, "env_kwargs.json"), "w") as f:
        json.dump(env_kwargs, f, indent=2)  # evaluate.py and render_video.py read this back to use the right configuration
    Vec = DummyVecEnv if args.dummy_vec else SubprocVecEnv
    venv = Vec([make_env(i, args.seed, env_kwargs) for i in range(args.n_envs)])
    stats_path = os.path.join(out_dir, "vecnormalize.pkl")
    if args.resume:
        src_stats = os.path.join(os.path.dirname(os.path.abspath(args.resume)), "vecnormalize.pkl")
        venv = VecNormalize.load(src_stats if os.path.exists(src_stats) else stats_path, venv)
        venv.training = True
    else:
        venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.99)

    try:
        import tensorboard  # noqa: F401
        tb = os.path.join(out_dir, "tb")
    except ImportError:
        tb = None

    if args.resume:
        model = PPO.load(args.resume, env=venv, device="cpu", tensorboard_log=tb)
    else:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=1024, batch_size=2048, n_epochs=5,
            learning_rate=lambda progress: 3e-4 * max(progress, 0.1),
            gamma=0.99, gae_lambda=0.95, clip_range=0.2, ent_coef=0.0,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256]), log_std_init=-1.0),
            seed=args.seed, device="cpu", tensorboard_log=tb, verbose=1,
        )

    levels = LEVELS_COURSE if args.scene == "course" else LEVELS_FIELD
    callbacks = [CheckpointCallback(
        save_freq=max(args.save_every // args.n_envs, 1), save_path=out_dir,
        name_prefix="ckpt", save_vecnormalize=True)]
    if args.no_curriculum:
        push, terrain, speed = levels[args.level]
        venv.env_method("set_difficulty", push, terrain, speed)
        print(f"[fixed] level {args.level}: push_max={push}, terrain={terrain}, speed={speed} m/s", flush=True)
    else:
        callbacks.append(CurriculumCallback(levels, start_level=args.start_level))

    try:
        model.learn(args.total_steps, callback=callbacks, reset_num_timesteps=not args.resume,
                    tb_log_name="ppo")
    finally:
        model.save(os.path.join(out_dir, "latest"))
        venv.save(stats_path)
        print("saved:", out_dir, flush=True)


if __name__ == "__main__":
    main()
