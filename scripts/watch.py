"""Live viewer: watch the policy walk in a MuJoCo window while training runs in another terminal.

Examples:
  python scripts/watch.py --run runs/run1
  python scripts/watch.py --run runs/run1 --push 40 --roughness 0.04

Follows the newest ckpt_*_steps.zip written by scripts/train.py (falls back to latest.zip). Close the window to quit.
To livestream to YouTube/Twitch, capture this window with OBS.
"""
import argparse
import glob
import os
import time

import mujoco
import mujoco.viewer

from biped_rl.env import BipedEnv
from evaluate import load_env_kwargs, load_policy, load_train_scene


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="training folder, e.g. runs/run1")
    p.add_argument("--seconds", type=float, default=15.0, help="episode length")
    p.add_argument("--push", type=float, default=0.0, help="impulse N*s at t=--push-time")
    p.add_argument("--push-time", type=float, default=4.0)
    p.add_argument("--roughness", type=float, default=0.0, help="scene field")
    p.add_argument("--challenge", default="stairs", help="scene course")
    p.add_argument("--level", type=float, default=0.5, help="scene course: difficulty [0, 1]")
    p.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    args = p.parse_args()

    run_stats = os.path.join(args.run, "vecnormalize.pkl")
    scene = load_train_scene(run_stats)
    env = BipedEnv(scene=scene, randomize=False, max_steps=int(args.seconds / 0.02) + 10, **load_env_kwargs(run_stats))

    def newest():
        """(model, stats) of the latest checkpoint; vecnormalize must match the checkpoint's step count."""
        for z in sorted(glob.glob(os.path.join(args.run, "ckpt_*_steps.zip")), key=os.path.getmtime, reverse=True):
            st = z.replace("ckpt_", "ckpt_vecnormalize_").replace(".zip", ".pkl")
            if os.path.exists(st):
                return z, st
        return os.path.join(args.run, "latest.zip"), run_stats

    cur, act, seed, alive = None, None, 0, True

    while alive:
        nxt = newest()
        if nxt != cur:  # new checkpoint from training
            while not os.path.exists(nxt[0]):  # waiting for the first checkpoint
                print("waiting for a checkpoint in", args.run, flush=True)
                time.sleep(10)
                nxt = newest()
            time.sleep(1)  # let the writer finish
            cur, act = nxt, load_policy(*nxt)
            print("loaded", os.path.basename(cur[0]), flush=True)
        opts = ({"challenge": args.challenge, "challenge_level": args.level} if scene == "course"
                else {"roughness": args.roughness, "on_reach": "resample"})
        if args.push > 0:
            opts.update(push_impulse=args.push, push_time=args.push_time)
        obs, _ = env.reset(seed=seed, options=opts)
        seed += 1
        # a fresh viewer per episode so it uploads the newly generated terrain
        with mujoco.viewer.launch_passive(env.model, env.data) as v:
            v.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            v.cam.trackbodyid = env.torso_id
            v.cam.distance, v.cam.azimuth, v.cam.elevation = 4.0, 120, -12
            while v.is_running():
                t0 = time.time()
                obs, r, term, trunc, info = env.step(act(obs))
                v.sync()
                if term or trunc:
                    break
                time.sleep(max(0.0, env.dt / args.speed - (time.time() - t0)))
            alive = v.is_running()


if __name__ == "__main__":
    main()
