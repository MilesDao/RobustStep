"""Evaluates a trained policy.

Two tests, each configuration run over many episodes with different seeds:
  1. push:    a horizontal push on the torso at t=3 s, random direction, increasing impulse (N*s).
              Reports survival rate vs. impulse and the impulse at which survival drops to 50%.
  2. terrain: walk to a goal 8 m away, increasing roughness.
              Reports goal-reach rate, fall rate, time to goal, cost of transport, action jerk.

Examples:
  python scripts/evaluate.py --model runs/run1/latest.zip --stats runs/run1/vecnormalize.pkl --label full --out eval/full
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from biped_rl.env import BipedEnv


def load_env_kwargs(stats_path):
    """Reads the environment config (kp_scale...) saved next to vecnormalize.pkl during training."""
    p = os.path.join(os.path.dirname(os.path.abspath(stats_path)), "env_kwargs.json")
    if os.path.exists(p):
        with open(p) as f:
            kw = json.load(f)
        return {"kp_scale": kw.get("kp_scale", 1.0), "height_scan": kw.get("height_scan", False)}
    return {}


def load_train_scene(stats_path):
    """Scene used during training ("field" or "course"), used as the default for scripts/render_video.py."""
    p = os.path.join(os.path.dirname(os.path.abspath(stats_path)), "env_kwargs.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f).get("scene", "field")
    return "field"


def load_policy(model_path, stats_path):
    model = PPO.load(model_path, device="cpu")
    kw = load_env_kwargs(stats_path)
    vn = VecNormalize.load(stats_path, DummyVecEnv([lambda: BipedEnv(**kw)]))
    vn.training = False
    vn.norm_reward = False

    def act(obs):
        a, _ = model.predict(vn.normalize_obs(obs), deterministic=True)
        return a
    return act


def run_episode(env, act, options, seed):
    obs, _ = env.reset(seed=seed, options=options)
    steps, jerk, info = 0, [], {}
    while True:
        obs, r, term, trunc, info = env.step(act(obs))
        jerk.append(info["smooth_abs"])
        steps += 1
        if term or trunc:
            break
    return dict(
        fell=bool(info["fell"]), success=bool(info["success"]), steps=steps,
        time=steps * env.dt, energy=info["total_energy"], path=info["path"],
        jerk=float(np.mean(jerk)), x_final=info.get("x", 0.0),
    )


def cot(env, energy, path):
    return energy / (env.total_mass * 9.81 * max(path, 1e-6))


def push_test(act, impulses, n, seed0, env_kwargs):
    env = BipedEnv(randomize=False, on_reach="ignore", max_steps=500, target_range=(14.0, 14.0), target_angle=0.0,
                   **env_kwargs)
    rows = []
    for J in impulses:
        res = [run_episode(env, act, {"push_impulse": J, "push_time": 3.0}, seed0 + i) for i in range(n)]
        rows.append(dict(impulse=float(J), survival=float(np.mean([not r["fell"] for r in res])), n=n))
        print(f"push {J:6.1f} N*s: survival {rows[-1]['survival']:.2f}", flush=True)
    return rows


def half_impulse(rows):
    xs = np.array([r["impulse"] for r in rows])
    ys = np.array([r["survival"] for r in rows])
    if ys.min() > 0.5:
        return None  # 50% point not found, need to raise the max impulse
    for i in range(1, len(xs)):
        if ys[i] <= 0.5 < ys[i - 1]:
            return float(xs[i - 1] + (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i]) * (xs[i] - xs[i - 1]))
    return float(xs[0])


def terrain_test(act, roughs, n, seed0, env_kwargs):
    env = BipedEnv(randomize=False, on_reach="terminate", max_steps=750, target_angle=0.0, **env_kwargs)
    rows = []
    for rg in roughs:
        res = [run_episode(env, act, {"roughness": rg, "target_dist": 8.0, "target_angle_fixed": 0.0}, seed0 + i)
               for i in range(n)]
        ok = [r for r in res if r["success"]]
        rows.append(dict(
            roughness=float(rg), success_rate=float(np.mean([r["success"] for r in res])),
            fall_rate=float(np.mean([r["fell"] for r in res])),
            time_to_goal=float(np.mean([r["time"] for r in ok])) if ok else None,
            cot=float(np.mean([cot(env, r["energy"], r["path"]) for r in ok])) if ok else None,
            jerk=float(np.mean([r["jerk"] for r in res])), n=n,
        ))
        print(f"terrain {rg:.2f} m: reached goal {rows[-1]['success_rate']:.2f}, fell {rows[-1]['fall_rate']:.2f}, "
              f"CoT {rows[-1]['cot']}", flush=True)
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--label", default="policy")
    p.add_argument("--out", default="outputs/eval")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--max-impulse", type=float, default=100.0)
    p.add_argument("--impulse-step", type=float, default=10.0)
    p.add_argument("--roughness", type=float, nargs="*", default=[0.0, 0.02, 0.04, 0.06, 0.08, 0.10])
    p.add_argument("--seed", type=int, default=1000)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    act = load_policy(args.model, args.stats)
    env_kwargs = load_env_kwargs(args.stats)
    print("environment config:", env_kwargs or "default")
    impulses = np.arange(0.0, args.max_impulse + 1e-6, args.impulse_step)
    push_rows = push_test(act, impulses, args.episodes, args.seed, env_kwargs)
    terrain_rows = terrain_test(act, args.roughness, args.episodes, args.seed, env_kwargs)
    half = half_impulse(push_rows)

    write_csv(os.path.join(args.out, "push.csv"), push_rows)
    write_csv(os.path.join(args.out, "terrain.csv"), terrain_rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(label=args.label, half_survival_impulse=half, push=push_rows, terrain=terrain_rows), f, indent=2)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot([r["impulse"] for r in push_rows], [r["survival"] for r in push_rows], marker="o", label=args.label)
    ax[0].axhline(0.5, color="gray", ls="--", lw=0.8)
    ax[0].set_xlabel("Impulse (N·s)")
    ax[0].set_ylabel("Survival rate")
    ax[0].set_ylim(-0.02, 1.02)
    ax[0].legend()
    ax[1].plot([r["roughness"] for r in terrain_rows], [r["success_rate"] for r in terrain_rows], marker="o", label=args.label)
    ax[1].set_xlabel("Roughness (m)")
    ax[1].set_ylabel("Goal-reach rate")
    ax[1].set_ylim(-0.02, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "curves.png"), dpi=150)
    print("impulse at 50% survival:", "not reached, raise --max-impulse" if half is None else f"{half:.1f} N*s")
    print("results saved to", args.out)


if __name__ == "__main__":
    main()
