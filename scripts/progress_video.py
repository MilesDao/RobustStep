"""Learning-progress video: one fixed-seed clip per checkpoint, with attempt count, practice time and stage overlaid.

Examples:
  python scripts/progress_video.py --runs runs/walk_scan runs/course_mixed --challenges stairs slope gaps stones \
      --every 2 --out g1_learning_progress.mp4
  python scripts/progress_video.py --runs runs/stand_a --out quick.mp4 --clip-seconds 3 --width 640 --height 360

Runs are played in the order given (e.g. a field run, then a course run resumed from it).
"""
import argparse
import os

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

from biped_rl import terrains
from biped_rl.env import BipedEnv
from evaluate import load_env_kwargs, load_policy, load_train_scene
from biped_rl.progress_data import attempts_at, format_steps, list_checkpoints, scalar_at
from render_video import make_renderer, rollout_frames
from train import DT, LEVELS_COURSE, LEVELS_FIELD

FONT_PATH = font_manager.findfont("DejaVu Sans")  # ships with matplotlib, has the ≈ and · glyphs


def _font(size):
    return ImageFont.truetype(FONT_PATH, size)


def draw_overlay(img, lines, banner=None, clock_red=False):
    """Text box top-left (last line red when clock_red), optional red banner centered."""
    h, w = img.shape[:2]
    im = Image.fromarray(img).convert("RGBA")
    layer = Image.new("RGBA", im.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    font = _font(max(14, h // 30))
    pad, gap = font.size // 2, font.size // 4
    heights = [d.textbbox((0, 0), s, font=font)[3] for s in lines]
    box_w = max(d.textlength(s, font=font) for s in lines) + 2 * pad
    box_h = sum(heights) + gap * (len(lines) - 1) + 2 * pad
    d.rectangle([pad, pad, pad + box_w, pad + box_h], fill=(0, 0, 0, 150))
    y = 2 * pad
    for i, (s, lh) in enumerate(zip(lines, heights)):
        red = clock_red and i == len(lines) - 1
        d.text((2 * pad, y), s, font=font, fill=(255, 80, 80, 255) if red else (255, 255, 255, 255))
        y += lh + gap
    if banner:
        big = _font(max(18, h // 12))
        d.text((w / 2, h / 2), banner, font=big, anchor="mm", fill=(255, 60, 60, 255),
               stroke_width=max(1, big.size // 12), stroke_fill=(0, 0, 0, 255))
    return np.asarray(Image.alpha_composite(im, layer).convert("RGB"))


def card(text, w, h, seconds, fps):
    """Black frames with centered white text."""
    im = Image.new("RGB", (w, h))
    ImageDraw.Draw(im).multiline_text((w / 2, h / 2), text, font=_font(max(16, h // 14)), anchor="mm",
                                      align="center", fill=(255, 255, 255))
    frame = np.asarray(im)
    return [frame] * int(round(seconds * fps))


def save_chart(attempts, survival, path):
    """PNG (1600x900) of survival time per clip vs attempt number."""
    fig, ax = plt.subplots(figsize=(16, 9), dpi=100)
    ax.plot(attempts, survival, "-o", color="#4c9be8", ms=7, lw=2)
    ax.set_xlim(0, max(attempts) * 1.05 + 1)
    ax.set_ylim(0, max(survival) * 1.1 + 0.1)
    ax.set_xlabel("Attempt #", fontsize=18)
    ax.set_ylabel("Seconds without falling (clip length caps it)", fontsize=18)
    ax.tick_params(labelsize=14)
    ax.grid(alpha=0.3)
    fig.savefig(path)
    plt.close(fig)


def stage_text(scene, level, kind):
    if level is None:
        return "Stage: unknown"
    if scene == "course":
        lv = LEVELS_COURSE[min(level, len(LEVELS_COURSE) - 1)][1]
        return f"{kind.capitalize()}: " + ", ".join(f"{k} {v}" for k, v in terrains.describe(kind, lv).items())
    push, terrain, speed = LEVELS_FIELD[min(level, len(LEVELS_FIELD) - 1)]
    if speed == 0:
        return "Stand still"
    if push == 0:
        return f"Walk {speed} m/s"
    return f"Pushes up to {push:g} N·s, rough {terrain * 100:g} cm"


def chain_offsets(run_dirs, last_steps):
    """(attempt offset, step offset) per run: totals of all earlier runs."""
    out, att, steps = [], 0, 0
    for i, run in enumerate(run_dirs):
        out.append((att, steps))
        if i < len(run_dirs) - 1:
            att += attempts_at(run, last_steps[i])[0]
            steps += last_steps[i]
    return out


def run_bases(cks):
    """Per run, the step count its own counter starts from: a run resumed with --resume keeps counting from its parent."""
    return [cks[i - 1][-1].steps if i > 0 and cks[i][0].steps > cks[i - 1][-1].steps else 0
            for i in range(len(cks))]


def render_clip(run, ck, scene, level, kind, args):
    """Roll out one checkpoint; returns ([(t, path, image)], info)."""
    act = load_policy(ck.model, ck.stats)
    env = BipedEnv(scene=scene, randomize=False, max_steps=int(args.clip_seconds / DT) + 10,
                   **load_env_kwargs(ck.stats))
    lv = level or 0
    if scene == "course":
        opts = {"challenge": kind, "challenge_level": LEVELS_COURSE[min(lv, len(LEVELS_COURSE) - 1)][1]}
    else:
        push, terrain, _ = LEVELS_FIELD[min(lv, len(LEVELS_FIELD) - 1)]
        opts = {"on_reach": "resample", "roughness": terrain}
        if push > 0:  # one visible push at half the level's maximum
            opts.update(push_impulse=0.5 * push, push_time=3.0)
    obs, _ = env.reset(seed=args.seed, options=opts)
    trace = [(0.0, 0.0)]

    def act_logged(o):  # remember distance walked at each policy step for the live readout
        trace.append((env.data.time, env._path))
        return act(o)

    renderer, cam = make_renderer(env, args.width, args.height)
    frames, info = rollout_frames(env, act_logged, obs, renderer, cam, args.clip_seconds, args.fps)
    renderer.close()
    trace.append((info["t"], info["path"]))
    times = np.array([t for t, _ in trace])
    out = [(t, trace[max(0, np.searchsorted(times, t, side="right") - 1)][1], img) for t, img in frames]
    return out, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True, help="run folders, played in this order")
    p.add_argument("--out", default="outputs/videos/progress.mp4")
    p.add_argument("--every", type=int, default=1, help="keep every K-th checkpoint (the last is always kept)")
    p.add_argument("--clip-seconds", type=float, default=8.0)
    p.add_argument("--challenges", nargs="+", default=["stairs", "slope", "gaps"], help="course clips cycle these")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--title", default="G1 learns to walk")
    args = p.parse_args()
    W, H, fps = args.width, args.height, args.fps

    cks = [list_checkpoints(r) for r in args.runs]
    for r, c in zip(args.runs, cks):
        if not c:
            raise SystemExit(f"no checkpoints in {r}")
    bases = run_bases(cks)
    local_last = [c[-1].steps - b for c, b in zip(cks, bases)]
    offsets = chain_offsets(args.runs, local_last)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = imageio.get_writer(args.out, fps=fps, macro_block_size=2)
    write = lambda frames: [writer.append_data(f) for f in frames]
    write(card(args.title, W, H, 3.0, fps))
    chart_att, chart_surv, clip_i, last_scene = [], [], 0, None
    for run, run_cks, base, (att_off, step_off) in zip(args.runs, cks, bases, offsets):
        scene = load_train_scene(run_cks[0].stats)
        chosen = run_cks[::args.every]
        if chosen[-1] is not run_cks[-1]:
            chosen.append(run_cks[-1])
        if scene != last_scene:
            write(card("Part 1 — Learning to walk" if scene == "field" else "Part 2 — Challenges", W, H, 2.0, fps))
            last_scene = scene
        prev_level = None
        for ck in chosen:
            local = ck.steps - base
            n, exact = attempts_at(run, local)
            n += att_off
            steps = step_off + local
            lvl = scalar_at(run, "curriculum/level", ck.steps)
            level = None if lvl is None else int(lvl)
            kind = args.challenges[clip_i % len(args.challenges)] if scene == "course" else None
            st = stage_text(scene, level, kind)
            if prev_level is not None and level != prev_level:
                write(card(f"Attempt #{n:,}:\n{st}", W, H, 2.0, fps))
            prev_level = level
            frames, info = render_clip(run, ck, scene, level, kind, args)
            head = [f"Attempt #{'' if exact else '≈'}{n:,}",
                    f"Practice: {steps * DT / 3600:.1f} h simulated",
                    f"Training steps: {format_steps(steps)}",
                    st if scene == "course" or st.startswith("Stage") else f"Stage: {st}"]
            for t, path, img in frames:
                writer.append_data(draw_overlay(img, head + [f"t = {t:.1f} s   distance {path:.1f} m"]))
            t, path, img = frames[-1]
            if info["fell"]:
                held = draw_overlay(img, head + [f"t = {t:.1f} s   distance {path:.1f} m"],
                                    banner=f"FELL at {t:.1f} s", clock_red=True)
                write([held] * (fps // 2))
            chart_att.append(n)
            chart_surv.append(info["t"])
            clip_i += 1
            print(f"{run} {format_steps(steps)}: attempt {n}, {'fell' if info['fell'] else 'ok'} at {info['t']:.1f}s",
                  flush=True)
    writer.close()
    chart = os.path.splitext(args.out)[0] + "_chart.png"
    save_chart(chart_att, chart_surv, chart)
    print("saved", args.out, "and", chart)


if __name__ == "__main__":
    main()
