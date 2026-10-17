"""Evaluates a policy on the corridor challenges (scene course).

Each challenge is run at several difficulty levels, each level over many episodes with different seeds. Reports:
goal-reach rate, fall rate, distance covered, time to goal, cost of transport,
and the difficulty level (with physical parameters) at which the goal-reach rate drops to 50%.

Examples:
  python evaluate_challenges.py --model runs/stairs1/latest.zip --stats runs/stairs1/vecnormalize.pkl \\
      --kinds stairs gaps slope --label stairs1 --out eval/stairs1
"""
import argparse
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import terrains
from biped_env import BipedEnv
from evaluate import cot, load_env_kwargs, load_policy, run_episode


def level_at_half(levels, success):
    """Difficulty level interpolated where the goal-reach rate drops through 0.5.

    Returns (level, status). Status: "interp" (crossing found), "all_above" (always above 50%, need to raise
    the max difficulty) or "all_below" (below 50% even at the easiest level, level is None).
    """
    ys = np.asarray(success)
    xs = np.asarray(levels)
    if ys.min() > 0.5:
        return None, "all_above"
    if ys[0] <= 0.5:
        return None, "all_below"
    for i in range(1, len(xs)):
        if ys[i] <= 0.5 < ys[i - 1]:
            return float(xs[i - 1] + (ys[i - 1] - 0.5) / (ys[i - 1] - ys[i]) * (xs[i] - xs[i - 1])), "interp"
    return None, "all_above"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--stats", required=True)
    p.add_argument("--label", default="policy")
    p.add_argument("--out", default="eval_challenges")
    p.add_argument("--kinds", nargs="*", default=terrains.KINDS)
    p.add_argument("--levels", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--payload", type=float, default=0.0, help="kg added to the upper body")
    p.add_argument("--friction", type=float, default=None, help="fixed friction multiplier")
    p.add_argument("--seed", type=int, default=2000)
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    act = load_policy(args.model, args.stats)
    kw = load_env_kwargs(args.stats)
    env = BipedEnv(scene="course", randomize=False, **kw)
    print("environment config:", kw or "default")

    rows, summary = [], {}
    for kind in args.kinds:
        succ = []
        for lv in args.levels:
            opts = {"challenge": kind, "challenge_level": lv}
            if args.payload > 0:
                opts["payload"] = args.payload
            if args.friction is not None:
                opts["friction"] = args.friction
            res = [run_episode(env, act, opts, args.seed + i) for i in range(args.episodes)]
            ok = [r for r in res if r["success"]]
            row = dict(
                kind=kind, level=lv, **{k: v for k, v in terrains.describe(kind, lv).items()},
                success_rate=float(np.mean([r["success"] for r in res])),
                fall_rate=float(np.mean([r["fell"] for r in res])),
                mean_x_final=float(np.mean([r["x_final"] for r in res])),
                time_to_goal=float(np.mean([r["time"] for r in ok])) if ok else None,
                cot=float(np.mean([cot(env, r["energy"], r["path"]) for r in ok])) if ok else None,
                jerk=float(np.mean([r["jerk"] for r in res])), n=args.episodes,
            )
            rows.append(row)
            succ.append(row["success_rate"])
            print(f"{kind:8s} level {lv:.2f} {terrains.describe(kind, lv)}: reached goal {row['success_rate']:.2f}, "
                  f"fell {row['fall_rate']:.2f}", flush=True)
        half, status = level_at_half(args.levels, succ)
        summary[kind] = dict(
            level_at_50=half, status=status,
            params_at_50=terrains.describe(kind, half) if half is not None else None,
        )
        if status == "interp":
            msg = f"{half:.2f} {terrains.describe(kind, half)}"
        elif status == "all_above":
            msg = "always above 50%, raise the max difficulty to find the limit"
        else:
            msg = "below 50% even at the easiest level"
        print(f"  -> {kind}: level at 50% = {msg}", flush=True)

    keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("kind", "level"), k))
    with open(os.path.join(args.out, "challenges.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(dict(label=args.label, config=kw, payload=args.payload, friction=args.friction,
                       summary=summary, rows=rows), f, indent=2, ensure_ascii=False)

    fig, ax = plt.subplots(figsize=(6.5, 4))
    for kind in args.kinds:
        pts = [(r["level"], r["success_rate"]) for r in rows if r["kind"] == kind]
        ax.plot(*zip(*pts), marker="o", label=kind)
    ax.axhline(0.5, color="gray", ls="--", lw=0.8)
    ax.set_xlabel("Difficulty level")
    ax.set_ylabel("Goal-reach rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "challenges.png"), dpi=150)
    print("results saved to", args.out)


if __name__ == "__main__":
    main()
