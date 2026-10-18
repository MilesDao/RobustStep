"""Records a demo video (offscreen render, camera follows the character, red arrow is the push force).

Examples:
  python scripts/render_video.py --model runs/run1/latest.zip --stats runs/run1/vecnormalize.pkl \
      --out demo_push.mp4 --push 40 --roughness 0.04 --seconds 12

If OpenGL errors occur: set MUJOCO_GL=egl (Linux/Kaggle) or MUJOCO_GL=glfw (Windows, default).
"""
import argparse
import os

import imageio
import mujoco
import numpy as np

from biped_rl.env import BipedEnv
from evaluate import load_env_kwargs, load_policy, load_train_scene


def add_arrow(scene, p0, p1, rgba, width=0.04):
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9),
                        np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width, p0, p1)
    scene.ngeom += 1


def make_renderer(env, width, height):
    """Offscreen renderer + tracking camera. Call after env.reset(), which generates the terrain."""
    renderer = mujoco.Renderer(env.model, height, width)
    mujoco.mjr_uploadHField(env.model, renderer._mjr_context, 0)  # terrain newly generated at reset
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = env.torso_id
    cam.distance, cam.azimuth, cam.elevation = 4.0, 120, -12
    return renderer, cam


def _render(env, renderer, cam):
    renderer.update_scene(env.data, camera=cam)
    f = env.current_push
    if np.linalg.norm(f) > 0:
        tp = env.data.xpos[env.torso_id] + np.array([0, 0, 0.3])
        add_arrow(renderer.scene, tp - f / np.linalg.norm(f) * 0.9, tp, [1, 0.1, 0.1, 1])
    return renderer.render()


def rollout_frames(env, act, obs, renderer, cam, seconds, fps):
    """Run the policy from `obs`; returns [(sim time, image)] and {"fell", "t", "path"}."""
    every = max(1, int(round(1.0 / (fps * env.dt))))
    frames = [(env.data.time, _render(env, renderer, cam))]
    fell, path = False, 0.0
    for i in range(1, int(round(seconds / env.dt)) + 1):
        obs, r, term, trunc, info = env.step(act(obs))
        path = info["path"]
        if i % every == 0 or term or trunc:
            frames.append((env.data.time, _render(env, renderer, cam)))
        if term or trunc:
            fell = term and not info["success"]
            break
    return frames, {"fell": bool(fell), "t": float(env.data.time), "path": float(path)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--out", default="outputs/videos/demo.mp4")
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--push", type=float, default=0.0, help="impulse N*s at t=--push-time")
    p.add_argument("--push-time", type=float, default=4.0)
    p.add_argument("--roughness", type=float, default=0.0, help="scene field")
    p.add_argument("--scene", choices=["field", "course"], default=None, help="defaults to env_kwargs.json")
    p.add_argument("--challenge", default=None, help="scene course: stairs, gaps, slope, ...")
    p.add_argument("--level", type=float, default=0.5, help="scene course: difficulty [0, 1]")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    act = load_policy(args.model, args.stats)
    kw = load_env_kwargs(args.stats)
    scene = args.scene or load_train_scene(args.stats)
    env = BipedEnv(scene=scene, randomize=False, max_steps=int(args.seconds / 0.02) + 10, **kw)
    if scene == "course":
        opts = {"challenge": args.challenge or "stairs", "challenge_level": args.level}
    else:
        opts = {"roughness": args.roughness, "on_reach": "resample"}
    if args.push > 0:
        opts.update(push_impulse=args.push, push_time=args.push_time)
    obs, _ = env.reset(seed=args.seed, options=opts)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    renderer, cam = make_renderer(env, args.width, args.height)
    frames, info = rollout_frames(env, act, obs, renderer, cam, args.seconds, args.fps)
    if info["fell"]:
        print(f"character fell at t={info['t']:.2f}s")
    frames = [img for _, img in frames]
    try:
        imageio.mimsave(args.out, frames, fps=args.fps)
    except Exception as e:  # save a gif if imageio-ffmpeg is missing
        out = args.out.rsplit(".", 1)[0] + ".gif"
        print("could not write mp4 (", e, "), saving", out)
        imageio.mimsave(out, frames[::2], fps=max(1, args.fps // 2))
        args.out = out
    print("saved", args.out, f"({len(frames)} frames)")


if __name__ == "__main__":
    main()
