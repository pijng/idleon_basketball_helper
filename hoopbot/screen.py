"""Locating the game canvas and grabbing frames in reference coordinates."""
from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np

from . import config as C

_logo = C.read_image(C.ASSETS / "logo.png", cv2.IMREAD_GRAYSCALE)
_exit = C.read_image(C.ASSETS / "exit.png", cv2.IMREAD_GRAYSCALE)


@dataclass
class Canvas:
    """Maps reference coordinates to physical image pixels: phys = origin + scale * ref."""
    ox: float
    oy: float
    scale: float

    def phys_rect(self, rect):
        x0, y0, x1, y1 = rect
        return (self.ox + x0 * self.scale, self.oy + y0 * self.scale,
                self.ox + x1 * self.scale, self.oy + y1 * self.scale)


def _best_match(gray, tmpl, scales):
    best = (-1.0, 1.0, (0, 0))
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] < 8 or t.shape[0] > gray.shape[0] or t.shape[1] > gray.shape[1]:
            continue
        _, v, _, loc = cv2.minMaxLoc(cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED))
        if v > best[0]:
            best = (v, s, loc)
    return best


def _locate_anchor(gray, tmpl, work=0.25):
    small = cv2.resize(gray, None, fx=work, fy=work, interpolation=cv2.INTER_AREA)
    v, s, loc = _best_match(small, tmpl, np.geomspace(0.15, 1.6, 48) * work)
    s_full = s / work
    # Refine at full resolution around the coarse hit.
    th, tw = int(tmpl.shape[0] * s_full), int(tmpl.shape[1] * s_full)
    x, y = int(loc[0] / work), int(loc[1] / work)
    pad = max(16, int(0.15 * max(th, tw)))
    x0, y0 = max(0, x - pad), max(0, y - pad)
    sub = gray[y0:y + th + pad, x0:x + tw + pad]
    v, s2, loc2 = _best_match(sub, tmpl, np.linspace(s_full * 0.93, s_full * 1.07, 15))
    return v, s2, (x0 + loc2[0], y0 + loc2[1])


def locate_canvas(img, min_score=0.7) -> Canvas | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    v1, s1, l1 = _locate_anchor(gray, _logo)
    v2, s2, l2 = _locate_anchor(gray, _exit)
    if min(v1, v2) < min_score:
        return None
    # Scale from anchor separation is far more precise than template scale.
    scale = (l2[1] - l1[1]) / (C.ANCHOR_EXIT[1] - C.ANCHOR_LOGO[1])
    if not 0.8 < scale / ((s1 + s2) / 2) < 1.25:
        return None
    ox = l1[0] - scale * C.ANCHOR_LOGO[0]
    oy = l1[1] - scale * C.ANCHOR_LOGO[1]
    return Canvas(ox, oy, scale)


def to_ref(img, canvas: Canvas, rect, out: float = 1.0):
    """Crop `rect` (reference coords) from a physical image, resampled to `out` x reference
    scale. Crop + resize (area filter when shrinking): cheaper and alias-free compared with
    a warp; the crop is off by under one physical pixel."""
    x0, y0, x1, y1 = rect
    w, h = max(1, int(round((x1 - x0) * out))), max(1, int(round((y1 - y0) * out)))
    s = canvas.scale
    a0, b0 = int(round(canvas.ox + x0 * s)), int(round(canvas.oy + y0 * s))
    a1, b1 = int(round(canvas.ox + x1 * s)), int(round(canvas.oy + y1 * s))
    H, W = img.shape[:2]
    pad = (max(0, -b0), max(0, b1 - H), max(0, -a0), max(0, a1 - W))
    crop = img[max(0, b0):min(H, b1), max(0, a0):min(W, a1)]
    if any(pad):
        crop = cv2.copyMakeBorder(crop, *pad, cv2.BORDER_REPLICATE)
    interp = cv2.INTER_AREA if w < crop.shape[1] else cv2.INTER_LINEAR
    return cv2.resize(crop, (w, h), interpolation=interp)


class ScreenSource:
    """Live capture through mss. Coordinates handed to mss/pynput are logical points;
    images may come back in physical pixels (dpr is measured, not assumed)."""

    def __init__(self, monitor_index: int | None = None):
        import mss
        self._sct = mss.MSS()
        mons = self._sct.monitors
        self.monitor_index = monitor_index
        self.monitor = mons[monitor_index] if monitor_index is not None else mons[0]
        self.canvas: Canvas | None = None
        self.dpr = 1.0

    def twin(self) -> "ScreenSource":
        """Same window, own capture handle (mss handles must not cross threads)."""
        s = ScreenSource(self.monitor_index)
        s.canvas, s.dpr = self.canvas, self.dpr
        return s

    def full(self):
        shot = self._sct.grab(self.monitor)
        img = np.asarray(shot)[:, :, :3]
        self.dpr = img.shape[1] / self.monitor["width"]
        return img

    def locate(self) -> Canvas | None:
        self.canvas = locate_canvas(self.full())
        return self.canvas

    def grab(self, rect, out: float = 1.0):
        """Return (image at `out` x reference scale, timestamp at capture midpoint)."""
        cv_ = self.canvas
        px0, py0, px1, py1 = cv_.phys_rect(rect)
        d = self.dpr
        region = {
            "left": int(np.floor(px0 / d)) + self.monitor["left"],
            "top": int(np.floor(py0 / d)) + self.monitor["top"],
        }
        region["width"] = int(np.ceil(px1 / d)) + self.monitor["left"] - region["left"]
        region["height"] = int(np.ceil(py1 / d)) + self.monitor["top"] - region["top"]
        t0 = time.perf_counter()
        img = np.asarray(self._sct.grab(region))[:, :, :3]
        t1 = time.perf_counter()
        # The region we got starts at a whole point; build a local canvas for it.
        lx = (region["left"] - self.monitor["left"]) * d
        ly = (region["top"] - self.monitor["top"]) * d
        local = Canvas(cv_.ox - lx, cv_.oy - ly, cv_.scale)
        return to_ref(img, local, rect, out), (t0 + t1) / 2

    def ref_to_point(self, x, y):
        """Reference coordinate -> logical screen point for the mouse."""
        cv_ = self.canvas
        return ((cv_.ox + x * cv_.scale) / self.dpr + self.monitor["left"],
                (cv_.oy + y * cv_.scale) / self.dpr + self.monitor["top"])


class FrameStream:
    """Captures one rect continuously in a background thread.

    On macOS a screen grab takes ~40 ms while processing a frame takes under 10 ms, so
    grabbing and processing in turn caps the bot at ~9 frames/s. Capturing in parallel
    lets the processing always pick up the newest frame. The capture calls release the
    GIL, so a thread is enough."""

    def __init__(self, src: ScreenSource, rect, out: float):
        import threading
        self.rect, self.out = rect, out
        self._src = src
        self._cond = threading.Condition()
        self._frame = None      # (img, t, seq)
        self._stop = False
        self._error = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            grabber = self._src.twin()
            seq = 0
            while not self._stop:
                img, t = grabber.grab(self.rect, self.out)
                seq += 1
                with self._cond:
                    self._frame = (img, t, seq)
                    self._cond.notify_all()
        except Exception as e:   # surfaced to the caller of next()
            with self._cond:
                self._error = e
                self._cond.notify_all()

    def next(self, after_seq: int, timeout: float = 1.0):
        """The newest frame with seq > after_seq: (img, t, seq)."""
        with self._cond:
            ok = self._cond.wait_for(
                lambda: self._error is not None or (self._frame is not None and self._frame[2] > after_seq),
                timeout)
            if self._error is not None:
                raise self._error
            if not ok:
                raise TimeoutError("screen capture stalled")
            return self._frame

    def close(self):
        self._stop = True


class VideoSource:
    """Offline frames from a recording, with true (VFR) timestamps."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(str(path))
        self.canvas: Canvas | None = None
        self.frame = None
        self.t = 0.0
        self.index = -1

    def next(self) -> bool:
        ok, f = self.cap.read()
        if not ok:
            return False
        self.frame = f
        self.t = self.cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        self.index += 1
        if self.canvas is None:
            self.canvas = locate_canvas(f)
            if self.canvas is None:
                raise RuntimeError("game canvas not found in first video frame")
        return True

    def grab(self, rect, out: float = 1.0):
        return to_ref(self.frame, self.canvas, rect, out), self.t


class ReplaySource:
    """Plays a recording in real time behind the live-capture interface (dry-run testing).

    A live capture runs slower than the game, so it never sees the same game frame twice
    unless the game froze. To keep that true here, a rect is never served the same video
    frame twice: the call waits for the next one."""

    def __init__(self, path, speed=1.0):
        self.video = VideoSource(path)
        self.video.next()
        self.canvas = self.video.canvas
        self.speed = speed
        self.start = None
        self.served = {}

    def locate(self):
        return self.canvas

    def grab(self, rect, out: float = 1.0):
        now = time.perf_counter()
        if self.start is None:
            self.start = now - self.video.t
        target = (now - self.start) * self.speed
        while self.video.t < target or self.served.get(rect) == self.video.index:
            if not self.video.next():
                raise EOFError("end of recording")
        self.served[rect] = self.video.index
        img, _ = self.video.grab(rect, out)
        return img, self.start + self.video.t / self.speed
