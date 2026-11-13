"""Game time and periodic motion of the platform and the hoop.

Game time. The game hitches: the picture freezes for 0.05-0.6 s. While frozen, the game
logic advances by at most DT_CAP (measured on footage2.mov: a 0.6 s freeze shifted the
platform's sine by ~0.25 s). Wall-clock fits across a hitch are therefore wrong, so every
sample is stamped with game time: wall time that only advances while frames change, and
by at most DT_CAP per change.

Motion. Platform and hoop coordinates are each either constant or
center + amp * sin(w g + phi) in game time.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

DT_CAP = 0.25          # s of game time a single frame can advance after a hitch
FROZEN_DIFF = 0.3      # mean |gray diff| vs the last changed frame below this = no change
W_RANGE = (0.4, 6.0)   # rad/s: periods from ~1 s to ~16 s
STILL_PX = 4.0         # a coordinate this flat is constant


class GameClock:
    def __init__(self):
        self.wall = None
        self.game = 0.0
        self.sig = None
        self.last_change = None   # wall time of the last frame that changed
        self.frozen_since = None

    def update(self, wall_t, signature) -> float:
        """Feed every captured frame (signature: small grayscale image). Returns game time."""
        sig = signature.astype(np.int16)
        if self.wall is None:
            self.wall, self.sig, self.last_change = wall_t, sig, wall_t
            return self.game
        changed = self.sig.shape != sig.shape or np.abs(sig - self.sig).mean() >= FROZEN_DIFF
        if changed:
            gap = wall_t - self.last_change
            # A slow capture (no identical frame in between) is not a hitch: full gap.
            self.game += min(gap, DT_CAP) if self.frozen_since is not None else gap
            self.last_change = wall_t
            self.frozen_since = None
            self.sig = sig   # compare with the last frame that changed, so slow motion adds up
        elif self.frozen_since is None:
            self.frozen_since = self.last_change
        self.wall = wall_t
        return self.game

    def now(self, wall_t) -> float:
        """Game time at wall_t, assuming the game keeps running from the last frame."""
        if self.last_change is None:
            return self.game
        return self.game + min(wall_t - self.last_change, DT_CAP)

    def frozen_for(self, wall_t) -> float:
        return 0.0 if self.frozen_since is None else wall_t - self.frozen_since

    def to_wall(self, g) -> float:
        """Wall time at which game time g arrives if nothing hitches."""
        return self.last_change + (g - self.game)


@dataclass
class Motion:
    center: float
    amp: float = 0.0
    w: float = 0.0
    phi: float = 0.0
    rms: float = 0.0
    n: int = 0

    @property
    def moving(self):
        return self.amp > 0

    def at(self, g):
        return self.center + self.amp * np.sin(self.w * np.asarray(g, float) + self.phi)

    def rate(self, g):
        return self.amp * self.w * np.cos(self.w * np.asarray(g, float) + self.phi)

    def describe(self):
        if not self.moving:
            return f"still at {self.center:.0f}"
        return (f"{self.center:.0f}±{self.amp:.0f}, period {2 * np.pi / self.w:.2f} s, "
                f"rms {self.rms:.1f}")


def fit_motion(g, v) -> Motion | None:
    g = np.asarray(g, float)
    v = np.asarray(v, float)
    if len(g) < 10:
        return None
    if np.percentile(v, 95) - np.percentile(v, 5) < STILL_PX:
        return Motion(float(np.median(v)), rms=float(np.std(v)), n=len(g))
    best = None
    for w in np.linspace(*W_RANGE, 117):
        X = np.stack([np.sin(w * g), np.cos(w * g), np.ones_like(g)], 1)
        c, *_ = np.linalg.lstsq(X, v, rcond=None)
        e = float(np.sum(np.minimum((X @ c - v) ** 2, 400.0)))
        if best is None or e < best[0]:
            best = (e, w, c)
    _, w, c = best
    p0 = [c[2], np.hypot(c[0], c[1]), w, np.arctan2(c[1], c[0])]
    r = least_squares(lambda p: p[0] + p[1] * np.sin(p[2] * g + p[3]) - v, p0,
                      loss="soft_l1", f_scale=5.0)
    c0, a, w, ph = r.x
    if a < 0:
        a, ph = -a, ph + np.pi
    res = c0 + a * np.sin(w * g + ph) - v
    return Motion(float(c0), float(a), float(w), float(ph % (2 * np.pi)),
                  float(np.sqrt(np.median(res ** 2))), len(g))

def fit_motion_2d(g, x, y) -> tuple[Motion, Motion] | None:
    """Both coordinates. When both move (an ellipse, either direction), they share one
    frequency: fitting them together pins the period much better than separately."""
    mx, my = fit_motion(g, x), fit_motion(g, y)
    if mx is None or my is None or not (mx.moving and my.moving):
        return (mx, my) if mx is not None and my is not None else None
    g = np.asarray(g, float)
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    best = None
    for w in np.linspace(*W_RANGE, 117):
        X = np.stack([np.sin(w * g), np.cos(w * g), np.ones_like(g)], 1)
        cx, *_ = np.linalg.lstsq(X, x, rcond=None)
        cy, *_ = np.linalg.lstsq(X, y, rcond=None)
        e = float(np.sum(np.minimum((X @ cx - x) ** 2, 400.0)) + np.sum(np.minimum((X @ cy - y) ** 2, 400.0)))
        if best is None or e < best[0]:
            best = (e, w, cx, cy)
    _, w, cx, cy = best
    p0 = [w, cx[2], np.hypot(cx[0], cx[1]), np.arctan2(cx[1], cx[0]),
          cy[2], np.hypot(cy[0], cy[1]), np.arctan2(cy[1], cy[0])]

    def resid(p):
        w, c1, a1, p1, c2, a2, p2 = p
        return np.concatenate([c1 + a1 * np.sin(w * g + p1) - x, c2 + a2 * np.sin(w * g + p2) - y])

    w, c1, a1, p1, c2, a2, p2 = least_squares(resid, p0, loss="soft_l1", f_scale=5.0).x
    out = []
    for c, a, ph, v in ((c1, a1, p1, x), (c2, a2, p2, y)):
        if a < 0:
            a, ph = -a, ph + np.pi
        res = c + a * np.sin(w * g + ph) - v
        out.append(Motion(float(c), float(a), float(w), float(ph % (2 * np.pi)),
                          float(np.sqrt(np.median(res ** 2))), len(g)))
    return out[0], out[1]



class Track:
    """Rolling history of one object's (x, y) in game time, refitted on demand."""

    def __init__(self, window_s: float):
        self.window = window_s
        self.samples: deque = deque()

    def add(self, g, x, y):
        self.samples.append((g, x, y))
        while self.samples and self.samples[0][0] < g - self.window:
            self.samples.popleft()

    def clear(self):
        self.samples.clear()

    def span(self):
        return self.samples[-1][0] - self.samples[0][0] if len(self.samples) > 1 else 0.0

    def last(self):
        return self.samples[-1] if self.samples else None

    def fit(self):
        if len(self.samples) < 10:
            return None
        S = np.array(self.samples)
        return fit_motion_2d(S[:, 0], S[:, 1], S[:, 2])
