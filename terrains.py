"""Generates heightfield challenge terrain for the long corridor (scene "course").

The corridor runs along the +x axis. Each challenge is a "segment" placed between flat-ground segments.
All functions work with height relative to the base ground (meters): 0 is ground, negative is a pit, positive is a ridge or step.

The difficulty `level` is in [0, 1]. `describe(kind, level)` returns the matching physical parameters
for the report.
"""
import numpy as np
from scipy.ndimage import gaussian_filter

PIT_DEPTH = 0.4
KINDS = ["rough", "slope", "stairs", "gaps", "stones", "hurdles", "beam"]


def describe(kind, level):
    """Physical parameters for a difficulty level (central value, random noise not included)."""
    lv = float(level)
    if kind == "flat":
        return {}
    if kind == "rough":
        return {"amplitude (cm)": round((0.02 + 0.08 * lv) * 100, 1)}
    if kind == "slope":
        return {"slope angle (deg)": round(3 + 12 * lv, 1)}
    if kind == "stairs":
        return {"step height (cm)": round((0.04 + 0.11 * lv) * 100, 1), "tread (cm)": 30}
    if kind == "gaps":
        return {"gap width (cm)": round((0.10 + 0.20 * lv) * 100, 1)}
    if kind == "stones":
        return {"gap between stones (cm)": round((0.10 + 0.20 * lv) * 100, 1),
                "stone length (cm)": round((0.50 - 0.15 * lv) * 100, 1)}
    if kind == "hurdles":
        return {"crossbar height (cm)": round((0.05 + 0.15 * lv) * 100, 1)}
    if kind == "beam":
        return {"beam width (cm)": round((0.50 - 0.30 * lv) * 100, 1)}
    raise ValueError(kind)


class _Grid:
    def __init__(self, xs, ys, level, rng):
        self.xs, self.ys, self.level, self.rng = xs, ys, level, rng
        self.H = np.zeros((len(ys), len(xs)))

    def cols(self, x0, x1):
        return (self.xs >= x0) & (self.xs < x1)

    def rows(self, y0, y1):
        return (self.ys >= y0) & (self.ys < y1)


def _jitter(g, scale=0.1):
    return 1.0 + g.rng.uniform(-scale, scale)


def seg_rough(g, x0):
    length = 4.0
    amp = (0.02 + 0.08 * g.level)
    fine = gaussian_filter(g.rng.standard_normal(g.H.shape), 3.0)
    coarse = gaussian_filter(g.rng.standard_normal(g.H.shape), 10.0)
    noise = 0.5 * fine / fine.std() + 0.5 * coarse / coarse.std()
    noise = np.clip(noise / 2.0, -1.0, 1.0)
    m = g.cols(x0, x0 + length)
    taper = np.clip(np.minimum(g.xs - x0, x0 + length - g.xs) / 0.6, 0.0, 1.0)  # smooth out at both ends
    g.H[:, m] += amp * noise[:, m] * taper[m]
    return x0 + length


def seg_slope(g, x0):
    ang = np.deg2rad((3 + 12 * g.level) * _jitter(g))
    ramp, plateau = 2.5, 1.0
    t = np.tan(ang)
    rise = ramp * t
    m1 = g.cols(x0, x0 + ramp)
    g.H[:, m1] += (g.xs[m1] - x0) * t
    x1 = x0 + ramp
    g.H[:, g.cols(x1, x1 + plateau)] += rise
    x2 = x1 + plateau
    m3 = g.cols(x2, x2 + ramp)
    g.H[:, m3] += rise - (g.xs[m3] - x2) * t
    return x2 + ramp


def seg_stairs(g, x0):
    n = int(g.rng.integers(4, 7))
    r = (0.04 + 0.11 * g.level) * _jitter(g, 0.05)
    tread, plateau = 0.30, 1.2
    for i in range(n):
        g.H[:, g.cols(x0 + i * tread, x0 + (i + 1) * tread)] += (i + 1) * r
    xp = x0 + n * tread
    g.H[:, g.cols(xp, xp + plateau)] += n * r
    xd = xp + plateau
    for i in range(n):
        g.H[:, g.cols(xd + i * tread, xd + (i + 1) * tread)] += (n - 1 - i) * r
    return xd + n * tread


def seg_gaps(g, x0):
    n, spacing = 3, 1.2
    x = x0 + 1.0
    for _ in range(n):
        w = (0.10 + 0.20 * g.level) * _jitter(g)
        g.H[:, g.cols(x, x + w)] = -PIT_DEPTH
        x += w + spacing
    return x - spacing + 1.0


def seg_stones(g, x0):
    n = 6
    ls = (0.50 - 0.15 * g.level)
    gap = (0.10 + 0.20 * g.level)
    xs0 = x0 + 0.5
    x_end = xs0 + n * ls + (n - 1) * gap
    g.H[:, g.cols(xs0, x_end)] = -PIT_DEPTH
    x = xs0
    for _ in range(n):
        jy = g.rng.uniform(-0.15, 0.15)
        rows = g.rows(jy - 0.6, jy + 0.6)
        g.H[np.ix_(rows, g.cols(x, x + ls))] = 0.0
        x += ls + gap
    return x_end + 0.5


def seg_hurdles(g, x0):
    n, thick, spacing = 3, 0.12, 1.3
    hb = (0.05 + 0.15 * g.level) * _jitter(g, 0.05)
    x = x0 + 1.0
    for _ in range(n):
        g.H[:, g.cols(x, x + thick)] += hb
        x += thick + spacing
    return x - spacing + 1.0


def seg_beam(g, x0):
    wb = 0.50 - 0.30 * g.level
    length = 3.0
    x1 = x0 + 0.5
    x2 = x1 + length
    cols = g.cols(x1, x2)
    g.H[:, cols] = -PIT_DEPTH
    rows = g.rows(-wb / 2, wb / 2)
    g.H[np.ix_(rows, cols)] = 0.0
    return x2 + 0.5


BUILDERS = {
    "rough": seg_rough, "slope": seg_slope, "stairs": seg_stairs, "gaps": seg_gaps,
    "stones": seg_stones, "hurdles": seg_hurdles, "beam": seg_beam,
}


def build_course(xs, ys, kind, level, rng, start_x=-10.0, flat_start=2.5, connector=1.5,
                 end_margin=3.0, max_repeats=3):
    """Builds the corridor. `kind` is one of KINDS, "flat", or "mixed" (random choice per segment).

    Returns (H, info). H has shape (len(ys), len(xs)), in meters relative to the ground.
    info: start_x, goal_x, segments = [(kind, x_start, x_end), ...].
    """
    g = _Grid(xs, ys, float(level), rng)
    x = start_x + flat_start
    x_limit = xs[-1] - end_margin
    segs = []
    if kind != "flat":
        while True:
            placed = False
            for _ in range(8 if kind == "mixed" else 1):  # try a few kinds until one fits the remaining space
                k = str(rng.choice(KINDS)) if kind == "mixed" else kind
                if k not in BUILDERS:
                    raise ValueError(f"invalid challenge kind: {k}")
                backup = g.H.copy()
                x_end = BUILDERS[k](g, x)
                if x_end <= x_limit:
                    segs.append((k, float(x), float(x_end)))
                    x = x_end + connector
                    placed = True
                    break
                g.H = backup
            if not placed or (kind != "mixed" and len(segs) >= max_repeats):
                break
        if not segs:
            raise ValueError(f"corridor too short for challenge '{kind}'")
    goal_x = float(min(xs[-1] - 1.5, x + 0.5)) if segs else float(x_limit)
    return g.H, {"start_x": start_x, "goal_x": goal_x, "segments": segs}
