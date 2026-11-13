"""Shot model, click planner and calibration.

The click fixes the whole flight:
  * The ball leaves the hand `release_delay` (~0.55 s) after the click.
  * What the ball inherits from the moving platform is taken from the platform's state
    `latency` seconds after the click (the jump): release point = platform position +
    offset + pos_k * platform velocity; launch velocity = constant + vel_k * platform
    velocity, separately per axis.
  * After release: x = xr + vx t, y = yr + vy t + g t^2 / 2 (game time).
  * A shot scores when the ball comes down through the rim level. Where the ball centre
    crosses it, relative to the hoop template's left edge (dx): clean swishes (+2) at
    dx ~100-165, backboard bank shots (+1) around 260-310, front rim/misses below ~90.
The planner scans click times and picks the middle of the longest stretch whose ball
crosses the rim within the target window. `fit` learns every constant above from logged
throws (live clicks plus the bundled video throws).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from . import config as C
from .motion import Motion

STEP_S = 0.004         # click-time grid
FLIGHT_S = 3.0         # longest flight considered
FLIGHT_STEP = 0.004


@dataclass
class Model:
    # Fitted on 22 live throws (macOS/CrossOver) + 6 recorded ones.
    latency: float = 0.34         # click -> platform state the ball inherits, s
    release_delay: float = 0.549  # click -> ball leaves the hand, s
    dx_release: float = 250.0     # release x - platform x
    dy_release: float = -666.0    # release y - platform y
    pos_kx: float = -0.28         # release point shift per platform velocity (s)
    pos_ky: float = -0.16
    vx: float = 1231.0
    k_vx: float = 0.44            # share of the platform's velocity the ball keeps
    vy0: float = -844.0
    k_vy: float = 0.34
    g: float = 2269.0
    target_dx: float = 145.0      # aim point: rim crossing x - hoop x (swishes: ~120-175)
    margin: float = 35.0          # half-width of the window we plan for (100-175 px all scored)
    dx_bias: float = 29.0         # measured rim crossing - path-model crossing (fitted)
    n_throws: int = 0

    def save(self, path: Path):
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "Model":
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in json.loads(Path(path).read_text()).items() if k in known})


@dataclass
class Prediction:
    g_click: np.ndarray     # game time of each candidate click
    dx: np.ndarray          # rim crossing x - hoop x (nan: never comes down through the rim level)
    g_arrive: np.ndarray


def launch(model: Model, plat, g_click):
    """Release game time, position and velocity for clicks at g_click (vectorised)."""
    gc = np.atleast_1d(np.asarray(g_click, float))
    gj = gc + model.latency
    vpx, vpy = plat[0].rate(gj), plat[1].rate(gj)
    xr = plat[0].at(gj) + model.dx_release + model.pos_kx * vpx
    yr = plat[1].at(gj) + model.dy_release + model.pos_ky * vpy
    return gc + model.release_delay, xr, yr, model.vx + model.k_vx * vpx, model.vy0 + model.k_vy * vpy


def predict(model: Model, plat: tuple[Motion, Motion], hoop: tuple[Motion, Motion],
            g_click) -> Prediction:
    """Where balls thrown at game times g_click cross the rim level (vectorised)."""
    gc = np.atleast_1d(np.asarray(g_click, float))
    gr, xr, yr, vxb, vyb = launch(model, plat, gc)

    t = np.arange(0.0, FLIGHT_S, FLIGHT_STEP)            # flight time grid
    G = gr[:, None] + t[None, :]
    bx = xr[:, None] + vxb[:, None] * t
    by = yr[:, None] + vyb[:, None] * t + 0.5 * model.g * t * t
    rim_y = hoop[1].at(G) + C.RIM_DY
    below = by >= rim_y
    descending = (vyb[:, None] + model.g * t) > 0
    hit = below & descending
    first = np.argmax(hit, axis=1)
    ok = hit[np.arange(len(gc)), first]
    rows = np.arange(len(gc))
    dx = np.where(ok, bx[rows, first] - hoop[0].at(G[rows, first]) + model.dx_bias, np.nan)
    return Prediction(gc, dx, np.where(ok, G[rows, first], np.nan))


@dataclass
class Plan:
    g_click: float
    dx: float
    window: float          # seconds of click time that land within the margin
    exact: bool            # False: nothing within the margin, best effort

    def describe(self):
        kind = f"window {self.window * 1000:.0f} ms" if self.exact else "best effort"
        return f"rim crossing dx {self.dx:.0f}, {kind}"


LOOKBACK_S = 0.3   # stretches already under way count if their middle is still ahead


def plan_click(model: Model, plat, hoop, g_from: float, horizon: float,
               near: float | None = None) -> Plan | None:
    """Best click at or after g_from. `near`: stay on the stretch around this click time
    (refining a committed plan) if it still exists."""
    gc = g_from - LOOKBACK_S + np.arange(0.0, horizon + LOOKBACK_S, STEP_S)
    p = predict(model, plat, hoop, gc)
    err = np.abs(p.dx - model.target_dx)
    good = np.nan_to_num(err, nan=1e9) <= model.margin
    if good.any():
        edges = np.flatnonzero(np.diff(np.concatenate(([0], good.astype(np.int8), [0]))))
        starts, ends = edges[::2], edges[1::2]
        mids = (starts + ends - 1) // 2
        ok = (gc[mids] >= g_from) & (ends < len(gc))
        if ok.any():
            idx = np.flatnonzero(ok)
            lengths = ends[idx] - starts[idx]
            k = None
            if near is not None:
                d = np.abs(gc[mids[idx]] - near)
                if d.min() < 0.15:
                    k = idx[int(np.argmin(d))]
            if k is None:
                # soonest stretch unless a later one is clearly longer
                k = idx[0] if lengths[0] >= 0.9 * lengths.max() else idx[int(np.argmax(lengths))]
            i = mids[k]
            return Plan(float(gc[i]), float(p.dx[i]), float((ends[k] - starts[k]) * STEP_S), True)
    ahead = gc >= g_from
    if np.isfinite(err[ahead]).any():
        i = int(np.flatnonzero(ahead)[np.nanargmin(err[ahead])])
        return Plan(float(gc[i]), float(p.dx[i]), 0.0, False)
    return None


# ---------------------------------------------------------------- calibration

HAND_LEAVE_PX = 40     # ball this far right of where it was held = released
HAND_DX = 245          # held ball x - platform x
MAX_FREEZE_S = 0.2     # throws with a longer game freeze during the flight are not fitted
MIN_EVAL_THROWS = 8    # measurable throws a held-out game needs for `fit` to judge a refit


BALL_MAX_SPEED = 2000  # px/s; a detection farther than this from the last one is something else
BALL_JUMP_SLACK = 80   # px allowed on top of that (detector jitter)
REVERSAL_PX = 60       # x going back by more than this = bounced (smaller dips are jitter)


def clean_path(path):
    """Ball detections in time order without stray blobs (e.g. a new score-ball icon in
    the HUD): each point must be reachable from the last accepted one."""
    P = sorted((tuple(p) for p in path), key=lambda p: p[0])
    out = []
    for p in P:
        if out:
            q = out[-1]
            if p[1:] == q[1:]:
                continue                       # frozen frame
            if np.hypot(p[1] - q[1], p[2] - q[2]) > BALL_JUMP_SLACK + BALL_MAX_SPEED * max(p[0] - q[0], 0.0):
                continue
        out.append(p)
    return np.array(out, float).reshape(-1, 3)


def free_flight(path, hoop_x_min):
    """Flight points between leaving the hand and reaching the rim (no bounces).

    `hoop_x_min`: the hoop's left edge, a number or a function of game time."""
    P = clean_path(path)
    if len(P) < 6:
        return None
    edge = hoop_x_min if callable(hoop_x_min) else (lambda g: hoop_x_min)
    x_hand = np.median(P[:3, 1])
    out, x_max = [], -np.inf
    for p in P:
        if p[1] < x_hand + HAND_LEAVE_PX and not out:
            continue
        if p[1] < x_max - REVERSAL_PX:
            break                      # bounced back
        if p[1] > edge(p[0]) - 20:
            break                      # at the rim
        if p[1] < x_max:
            continue                   # small dip: detector jitter
        out.append(p)
        x_max = p[1]
    return np.array(out) if len(out) >= 6 else None


def _flight_data(throws):
    """(throw, platform fits, free flight, release game time) for usable throws."""
    from .motion import fit_motion
    out = []
    for t in throws:
        if len(t.platform) < 20 or len(t.hoop) < 5:
            continue
        if t.extra.get("max_freeze", 0.0) > MAX_FREEZE_S:
            continue   # a hitch around the click shifts its game time
        S = np.asarray(t.platform, float)
        H = np.asarray(t.hoop, float)
        plat = (fit_motion(S[:, 0], S[:, 1]), fit_motion(S[:, 0], S[:, 2]))
        hx = fit_motion(H[:, 0], H[:, 1])
        if plat[0] is None or plat[1] is None or hx is None:
            continue
        F = free_flight(t.flight, hx.center - hx.amp)
        if F is None:
            continue
        qx = np.polyfit(F[:, 0], F[:, 1], 1)
        g_ref = t.g_click if t.g_click is not None else F[0, 0]
        hand = plat[0].at(g_ref) + HAND_DX
        out.append((t, plat, F, float((hand - qx[1]) / qx[0])))
    return out


def fit_model(throws, model: Model | None = None, log=print) -> Model:
    """Calibrate on whole flight paths.

    Live throws know their click time; recorded ones (extra["source"] == "video") don't, so
    their click is put `release_delay` before the measured release."""
    model = model or Model()
    data = _flight_data(throws)
    live = [d for d in data if d[0].g_click is not None]
    if len(live) < 10:
        log(f"only {len(live)} usable live throws; model unchanged")
        return model
    model.release_delay = float(np.median([gr - t.g_click for t, _, _, gr in live]))
    clicks = [t.g_click if t.g_click is not None else gr - model.release_delay for t, _, _, gr in data]
    vy_seen = [plat[1].rate(c + model.latency) for (_, plat, _, _), c in zip(data, clicks)]
    log(f"{len(live)} live + {len(data) - len(live)} recorded throws; ball leaves the hand "
        f"{model.release_delay:.3f} s after the click; platform vy at the jump "
        f"{min(vy_seen):.0f}..{max(vy_seen):.0f}")

    names = ["latency", "dx_release", "vx", "k_vx", "dy_release", "vy0", "k_vy", "g",
             "pos_ky", "pos_kx"]

    def resid(p):
        m = Model(**{**asdict(model), **dict(zip(names, p))})
        out = []
        for (t, plat, F, _), c in zip(data, clicks):
            gr, xr, yr, vx, vy = launch(m, plat, c)
            dt = F[:, 0] - gr[0]
            out.append(xr[0] + vx[0] * dt - F[:, 1])
            out.append(yr[0] + vy[0] * dt + 0.5 * m.g * dt * dt - F[:, 2])
        return np.concatenate(out)

    p0 = [getattr(model, n) for n in names]
    best = None
    for lat in (0.0, 0.15, 0.3, 0.45):   # the latency trades off against the velocity terms
        p0[0] = lat
        r = least_squares(resid, p0, loss="soft_l1", f_scale=15.0)
        if best is None or r.cost < best.cost:
            best = r
    for n, v in zip(names, best.x):
        setattr(model, n, float(v))
    log(f"launch: platform state at +{model.latency:.3f} s; release x = plat + {model.dx_release:.0f} "
        f"{model.pos_kx:+.2f}*vx_p, y = plat {model.dy_release:+.0f} {model.pos_ky:+.2f}*vy_p; "
        f"vx = {model.vx:.0f} {model.k_vx:+.2f}*vx_p, vy = {model.vy0:.0f} {model.k_vy:+.2f}*vy_p; "
        f"g {model.g:.0f}; path error median {np.median(np.abs(best.fun)):.1f} px")

    # The path fit weighs the whole flight; where the ball actually meets the rim can still
    # differ a little. Measure that against the hoop seen during the flight.
    from .motion import fit_motion_2d
    diffs = []
    for t, plat, _, _ in live:
        act = rim_crossing(t.flight, t.extra.get("hoop_after"), t.hoop)
        if act is None:
            continue
        H = np.asarray(t.hoop, float)
        hoop = fit_motion_2d(H[:, 0], H[:, 1], H[:, 2])
        pred = predict(model, plat, hoop, [t.g_click]).dx[0] - model.dx_bias
        if np.isfinite(pred):
            diffs.append(act - pred)
    if len(diffs) >= 10:
        model.dx_bias = float(np.median(diffs))
        log(f"rim crossing bias {model.dx_bias:+.0f} px (n={len(diffs)})")
    swish = [rim_crossing(t.flight, t.extra.get("hoop_after"), t.hoop) for t, _, _, _ in live
             if t.points == 2]
    swish = [s for s in swish if s is not None]
    if len(swish) >= 10:
        model.target_dx = float(np.median(swish))
        log(f"target: rim crossing dx {model.target_dx:.0f} (median of {len(swish)} swishes)")
    model.n_throws = len(data)
    return model


NEAR_BALL_PX = 220     # hoop samples with the ball this close are distorted (ball over rim)
HOOP_RELOCATED_PX = 100
MAX_EXTRAPOLATE_S = 0.5


def rim_crossing(path, hoop_after, hoop_before=()):
    """Where a recorded ball came down through the rim level, relative to the hoop's left
    edge; None if it can't be measured.

    The hoop is taken from a fit of its path (before and during the flight), ignoring
    frames where the ball overlaps it and everything after it jumped to a new spot; the
    ball from a parabola through its free flight, extrapolated at most MAX_EXTRAPOLATE_S."""
    from .motion import Motion, fit_motion_2d
    if not hoop_after or len(path) < 6:
        return None
    P = clean_path(path)
    if len(P) < 6:
        return None
    H = np.array(sorted(list(map(tuple, hoop_before)) + list(map(tuple, hoop_after))), float)
    bx = np.interp(H[:, 0], P[:, 0], P[:, 1], left=np.nan, right=np.nan)
    by = np.interp(H[:, 0], P[:, 0], P[:, 2], left=np.nan, right=np.nan)
    near = np.hypot(bx - (H[:, 1] + 130), by - H[:, 2]) < NEAR_BALL_PX
    H = H[~near]
    after = H[:, 0] >= np.asarray(hoop_after, float)[0, 0]
    step = np.r_[0, np.abs(np.diff(H[:, 1])) + np.abs(np.diff(H[:, 2]))]
    jumped = np.flatnonzero(after & (step > HOOP_RELOCATED_PX))
    if len(jumped):
        H = H[: jumped[0]]
    if len(H) < 10:
        return None
    hoop = fit_motion_2d(H[:, 0], H[:, 1], H[:, 2]) if np.ptp(H[:, 1]) + np.ptp(H[:, 2]) > 8 else \
        (Motion(float(np.median(H[:, 1]))), Motion(float(np.median(H[:, 2]))))
    if hoop is None:
        return None
    F = free_flight(P, lambda g: hoop[0].at(g))
    if F is None:
        return None
    qx = np.polyfit(F[:, 0], F[:, 1], 1)
    qy = np.polyfit(F[:, 0], F[:, 2], 2)
    if qy[0] <= 0:
        return None
    g = np.arange(F[0, 0], F[-1, 0] + MAX_EXTRAPOLATE_S, 0.002)
    rim = hoop[1].at(g) + C.RIM_DY
    down = (np.polyval(qy, g) >= rim) & (np.polyval(np.polyder(qy), g) > 0)
    if not down.any():
        return None
    i = int(np.argmax(down))
    return float(np.polyval(qx, g[i]) - hoop[0].at(g[i]))


# ---------------------------------------------------------------- honest evaluation

def sessions(throws):
    """Live throws grouped by game session (each session's game clock starts at 0)."""
    out, prev = [], None
    for t in throws:
        if t.g_click is None:
            continue
        if prev is None or t.g_click < prev:
            out.append([])
        out[-1].append(t)
        prev = t.g_click
    return out


@dataclass
class Score:
    n: int
    bias: float          # median (predicted - measured) rim crossing, px
    spread: float        # robust std of that, px
    within: float        # share within the planner's margin

    def describe(self):
        return (f"{self.n} throws, bias {self.bias:+.0f} px, spread ±{self.spread:.0f} px, "
                f"{self.within:.0%} within the aim window")


def evaluate(model: Model, throws) -> Score | None:
    """How well `model` predicts where the logged balls really crossed the rim.

    Needs throws logged with the hoop's path during the flight (extra["hoop_after"])."""
    from .motion import fit_motion, fit_motion_2d
    errs = []
    for t in throws:
        if t.g_click is None or len(t.hoop) < 10 or len(t.platform) < 20:
            continue
        act = rim_crossing(t.flight, t.extra.get("hoop_after"), t.hoop)
        if act is None:
            continue
        S = np.asarray(t.platform, float)
        H = np.asarray(t.hoop, float)
        plat = (fit_motion(S[:, 0], S[:, 1]), fit_motion(S[:, 0], S[:, 2]))
        hoop = fit_motion_2d(H[:, 0], H[:, 1], H[:, 2])
        if None in plat or hoop is None:
            continue
        d = float(predict(model, plat, hoop, [t.g_click]).dx[0]) - act
        if np.isfinite(d):
            errs.append(d)
    if not errs:
        return None
    e = np.array(errs)
    med = float(np.median(e))
    return Score(len(e), med, float(1.4826 * np.median(np.abs(e - med))),
                 float(np.mean(np.abs(e) <= model.margin)))
