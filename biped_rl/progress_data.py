"""Reads a training run folder for the progress video: checkpoints, attempt counts and curriculum scalars."""
import csv
import glob
import os
import re
from dataclasses import dataclass

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


@dataclass
class Checkpoint:
    steps: int
    model: str
    stats: str


def list_checkpoints(run_dir):
    """Checkpoints with a matching VecNormalize file, oldest first."""
    out = []
    for z in glob.glob(os.path.join(run_dir, "ckpt_*_steps.zip")):
        m = re.search(r"ckpt_(\d+)_steps\.zip$", z)
        st = os.path.join(run_dir, f"ckpt_vecnormalize_{m.group(1)}_steps.pkl")
        if os.path.exists(st):  # the stats file is written just after the zip
            out.append(Checkpoint(int(m.group(1)), z, st))
    return sorted(out, key=lambda c: c.steps)


def _tb_points(run_dir, tag):
    pts = []
    for f in glob.glob(os.path.join(run_dir, "tb", "**", "events.out.tfevents.*"), recursive=True):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        if tag in ea.Tags()["scalars"]:
            pts += [(e.step, e.value) for e in ea.Scalars(tag)]
    return sorted(pts)


def attempts_at(run_dir, steps):
    """(episodes finished before `steps`, exact). Exact from Monitor CSVs, else estimated from TensorBoard."""
    lengths = []
    for f in glob.glob(os.path.join(run_dir, "monitor_*.monitor.csv")):
        with open(f, newline="") as fh:
            next(fh)  # JSON header line
            lengths.append([int(r["l"]) for r in csv.DictReader(fh) if r["l"] and r["l"].isdigit()])
    # CSVs erased by an older resume cover fewer steps than the checkpoint; they would undercount
    if lengths and sum(map(sum, lengths)) >= 0.9 * steps:
        per_env = steps / len(lengths)
        n = sum(int(np.sum(np.cumsum(ls) <= per_env)) for ls in lengths)
        return n, True
    est, prev = 0.0, 0
    for step, ep_len in _tb_points(run_dir, "rollout/ep_len_mean"):
        if step > steps:
            break
        est += (step - prev) / max(ep_len, 1.0)
        prev = step
    return int(round(est)), False


def scalar_at(run_dir, tag, steps):
    """Last logged value of `tag` at or before `steps`, None if never logged."""
    val = None
    for step, v in _tb_points(run_dir, tag):
        if step > steps:
            break
        val = v
    return val


def format_steps(steps):
    return f"{steps / 1e6:.2f}".rstrip("0").rstrip(".") + "M"
