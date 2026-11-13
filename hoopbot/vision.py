"""Detectors.

`read_scene` takes the field (C.FIELD) at WORK x reference scale; other readers take
reference-scale crops.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C

WORK = 0.5     # the field is analysed at half resolution
COARSE = 0.5   # template search runs at WORK * COARSE first, then is refined locally


def _template(name, scale, rows=None):
    t = C.read_image(C.ASSETS / name, cv2.IMREAD_GRAYSCALE)
    if rows is not None:
        t = t[:rows]
    return cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


# The hoop can sit so low that its net is cut off by the bottom of the screen; the rim and
# the top of the net (first HOOP_ROWS rows of the template) are enough to find it, down to
# a rim at y ~1638.
HOOP_ROWS = 70


_PLAT = _template("platform.png", WORK)
_HOOP = _template("hoop.png", WORK, HOOP_ROWS)
_PLAT_C = _template("platform.png", WORK * COARSE)
_HOOP_C = _template("hoop.png", WORK * COARSE, HOOP_ROWS)
_GAME_OVER = C.read_image(C.ASSETS / "gameover.png", cv2.IMREAD_GRAYSCALE)
_LOGO = C.read_image(C.ASSETS / "logo.png", cv2.IMREAD_GRAYSCALE)


@dataclass
class Scene:
    t: float
    plat: tuple[float, float]
    plat_score: float
    hoop: tuple[float, float]
    hoop_score: float
    ball: tuple[float, float] | None   # flying ball centre, if any
    ball_mask: np.ndarray               # raw orange mask (for static removal)
    gray: np.ndarray                    # grayscale field at WORK scale


TRACK_PX = 220        # local search radius around the last position, reference px
TRACK_MIN_SCORE = 0.6 # below this the local search is not trusted: search everywhere


def _find(gray, gray_coarse, tmpl, tmpl_coarse, hint=None):
    """Template top-left in reference coords and match score.

    With `hint` (last top-left), searches only around it, which is ~10x cheaper; falls
    back to a coarse-to-fine search of the whole field if that fails."""
    if hint is not None:
        hx = int((hint[0] - C.FIELD[0]) * WORK)
        hy = int((hint[1] - C.FIELD[1]) * WORK)
        r = int(TRACK_PX * WORK)
        x0, y0 = max(0, hx - r), max(0, hy - r)
        sub = gray[y0:hy + tmpl.shape[0] + r, x0:hx + tmpl.shape[1] + r]
        if sub.shape[0] >= tmpl.shape[0] and sub.shape[1] >= tmpl.shape[1]:
            _, v, _, l = cv2.minMaxLoc(cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED))
            if v >= TRACK_MIN_SCORE:
                return (C.FIELD[0] + (x0 + l[0]) / WORK, C.FIELD[1] + (y0 + l[1]) / WORK), float(v)
    if callable(gray_coarse):
        gray_coarse = gray_coarse()
    _, _, _, l = cv2.minMaxLoc(cv2.matchTemplate(gray_coarse, tmpl_coarse, cv2.TM_CCOEFF_NORMED))
    x, y = int(l[0] / COARSE), int(l[1] / COARSE)
    pad = 6
    x0, y0 = max(0, x - pad), max(0, y - pad)
    sub = gray[y0:y + tmpl.shape[0] + pad, x0:x + tmpl.shape[1] + pad]
    if sub.shape[0] < tmpl.shape[0] or sub.shape[1] < tmpl.shape[1]:
        return (0.0, 0.0), 0.0
    _, v, _, l = cv2.minMaxLoc(cv2.matchTemplate(sub, tmpl, cv2.TM_CCOEFF_NORMED))
    return (C.FIELD[0] + (x0 + l[0]) / WORK, C.FIELD[1] + (y0 + l[1]) / WORK), float(v)


def orange_mask(img_small):
    hsv = cv2.cvtColor(img_small, cv2.COLOR_BGR2HSV)
    return (hsv[..., 0] >= 4) & (hsv[..., 0] <= 20) & (hsv[..., 1] > 140) & (hsv[..., 2] > 140)


def _blank(mask, rect_ref):
    x0, y0, x1, y1 = rect_ref
    ox, oy = C.FIELD[0], C.FIELD[1]
    r0, r1 = max(0, int((y0 - oy) * WORK)), max(0, int((y1 - oy) * WORK))
    c0, c1 = max(0, int((x0 - ox) * WORK)), max(0, int((x1 - ox) * WORK))
    mask[r0:r1, c0:c1] = False


def read_scene(small, t, static=None, hints=(None, None)) -> Scene:
    """Platform, hoop and flying ball from the field at WORK scale.

    `static`: orange pixels that don't move (HUD, decorations); ignored for the ball.
    `hints`: last (platform, hoop) positions, for the cheap local search."""
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    cache = []

    def coarse():   # only needed when a local search fails
        if not cache:
            cache.append(cv2.resize(gray, None, fx=COARSE, fy=COARSE, interpolation=cv2.INTER_AREA))
        return cache[0]

    plat, pv = _find(gray, coarse, _PLAT, _PLAT_C, hints[0])
    hoop, hv = _find(gray, coarse, _HOOP, _HOOP_C, hints[1])
    raw = orange_mask(small)
    m = raw & ~static if static is not None and static.shape == raw.shape else raw.copy()
    px, py = plat
    dx0, dy0, dx1, dy1 = C.PLAYER_BOX
    _blank(m, (px + dx0, py + dy0, px + dx1, py + dy1))
    hx, hy = hoop
    rx0, ry0, rx1, ry1 = C.RIM_BOX
    _blank(m, (hx + rx0, hy + ry0, hx + rx1, hy + ry1))
    n, _, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    ball, best = None, 0.0
    lo, hi = C.BALL_AREA
    for j in range(1, n):
        area = st[j, 4] / (WORK * WORK)
        aspect = st[j, 2] / max(st[j, 3], 1)
        if lo < area < hi and 0.6 < aspect < 1.6 and area > best:
            best = area
            ball = (C.FIELD[0] + cen[j][0] / WORK, C.FIELD[1] + cen[j][1] / WORK)
    return Scene(t, plat, pv, hoop, hv, ball, raw, gray)


def logo_visible(img) -> bool:
    """`img` = C.LOGO_BOX: is the "SWISHY HOOPS" title still where the game was found?"""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, v, _, _ = cv2.minMaxLoc(cv2.matchTemplate(g, _LOGO, cv2.TM_CCOEFF_NORMED))
    return v > 0.7


def game_over(banner_img) -> bool:
    g = cv2.cvtColor(banner_img, cv2.COLOR_BGR2GRAY)
    if g.shape[0] < _GAME_OVER.shape[0] or g.shape[1] < _GAME_OVER.shape[1]:
        return False
    _, v, _, _ = cv2.minMaxLoc(cv2.matchTemplate(g, _GAME_OVER, cv2.TM_CCOEFF_NORMED))
    return v > 0.8


# ---------------------------------------------------------------- score

def _load_glyphs():
    out = []
    for p in sorted((C.ASSETS / "glyphs").glob("*.png")):
        g = C.read_image(p, cv2.IMREAD_GRAYSCALE) > 127
        out.append((p.stem.split("_")[0], g))
    return out


_NORM = (22, 28)


def _norm(g):
    return cv2.resize(g.astype(np.float32), _NORM, interpolation=cv2.INTER_AREA)


_GLYPHS = [(ch, _norm(g), g.shape[1] / g.shape[0]) for ch, g in _load_glyphs()]


def _classify(g):
    gn, ar = _norm(g), g.shape[1] / g.shape[0]
    d, ch = min((np.abs(gn - t).mean() + 0.3 * abs(ar - a), ch) for ch, t, a in _GLYPHS)
    return ch, float(d)


def read_score(score_img):
    """`score_img` = C.SCORE_TEXT. Reads the run of digit-sized glyphs after "Score:"."""
    v = cv2.cvtColor(score_img, cv2.COLOR_BGR2HSV)
    m = ((v[..., 2] > 170) & (v[..., 1] < 80)).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = sorted(tuple(st[j]) for j in range(1, n) if st[j, 4] > 15)
    digits, last_x = "", None
    for x, y, w, h, _ in comps:
        if x < C.SCORE_DIGITS_X or h < 25:
            continue
        if last_x is not None and x - last_x > 20:
            break   # a star, not the next digit
        ch, d = _classify(m[y:y + h, x:x + w] > 0)
        if not ch.isdigit() or d > 0.2:
            return None
        digits += ch
        last_x = x + w
    return int(digits) if digits else None
