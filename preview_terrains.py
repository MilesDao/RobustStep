"""Renders preview images of the challenges at several difficulty levels (no policy needed).

  python preview_terrains.py --out previews
  python preview_terrains.py --kinds stairs gaps --levels 0 0.5 1

One PNG per challenge, one row per difficulty level. The robot stands at the start, the red sphere is the goal.
Linux and Kaggle need MUJOCO_GL=egl.
"""
import argparse
import os

import imageio
import mujoco
import numpy as np

import terrains
from biped_env import BipedEnv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="previews")
    p.add_argument("--kinds", nargs="*", default=terrains.KINDS + ["mixed"])
    p.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.5, 1.0])
    p.add_argument("--width", type=int, default=1500)
    p.add_argument("--height", type=int, default=380)
    p.add_argument("--seed", type=int, default=3)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    env = BipedEnv(scene="course", randomize=False)
    renderer = mujoco.Renderer(env.model, args.height, args.width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0.0, 0.0, 0.8]
    cam.distance, cam.azimuth, cam.elevation = 15.0, 80.0, -32.0

    for kind in args.kinds:
        rows = []
        for lv in args.levels:
            env.reset(seed=args.seed, options={"challenge": kind, "challenge_level": lv})
            mujoco.mjr_uploadHField(env.model, renderer._mjr_context, 0)
            renderer.update_scene(env.data, camera=cam)
            rows.append(renderer.render())
            print(f"{kind:8s} level={lv:.2f}  {terrains.describe(kind, lv) if kind != 'mixed' else ''}"
                  f"  segments: {[s[0] for s in env._course['segments']]}")
        imageio.imwrite(os.path.join(args.out, f"{kind}.png"), np.concatenate(rows, axis=0))
    print("saved images to", args.out)


if __name__ == "__main__":
    main()
