"""Live play: watch, plan, click, learn."""
from __future__ import annotations

import logging
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config as C
from . import records
from . import vision as V
from .motion import GameClock, Track
from .physics import Model, plan_click, rim_crossing
from .records import Throw
from .screen import FrameStream, ReplaySource, ScreenSource

log = logging.getLogger("hoopbot")

PLATFORM_WINDOW_S = 8.0     # platform history used for its fit (it moves continuously)
HOOP_WINDOW_S = 5.0         # a bit over one period of a moving hoop; short, in case it turns
HOOP_FIT_RMS = 10.0         # a moving hoop's fit must follow its samples this closely
MIN_PLATFORM_S = 2.5        # observe this much platform motion before the first throw
MIN_HOOP_STILL_S = 0.6      # hoop must be settled this long after it jumps
MIN_HOOP_MOVING_S = 3.0     # ... and observed this long if it moves
HOOP_JUMP_PX = 40           # still hoop: position change that means "it moved to a new spot"
HOOP_RELOCATE_PX = 60       # moving hoop: distance from its fitted path that means the same
HOOP_MAX_SPEED = 600        # px/s (|dx|+|dy|) a moving hoop never exceeds; faster = relocation
MIN_LEAD_S = 0.25           # never plan a click sooner than this
REFINE_BEFORE_S = 0.15      # last re-plan this long before the click
HITCH_S = 0.06              # frozen this long right before a click = skip this chance
CLICK_HOLD_S = 0.03
FLIGHT_MAX_S = 3.5
FLIGHT_GONE_S = 0.8         # ball unseen this long (game time) after flying = flight over
NO_FLIGHT_S = 1.5           # no ball seen this long after a click = the game ignored it
SETTLE_S = 0.6              # after a flight, wait before reading the score / planning
IDLE_TIMEOUT_S = 30.0
HEARTBEAT_S = 2.0
# Best possible rim crossing must be in this band to throw at all. Measured on 85 live
# throws: 100-175 px all scored (46 of 57 for +2); 175-225 px 7 of 9 scored (off the
# backboard); beyond that half missed, and short of ~95 px nearly all missed.
REACH_BAND = (95, 225)
TROPHY_SCORE = 40
CURSOR_PARK = (1500, 1650)  # bottom middle of the field


def setup_logging(debug: bool, log_dir: Path) -> Path | None:
    log.setLevel(logging.DEBUG)
    log.handlers.clear()
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if debug else logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(message)s", "%H:%M:%S"))
    log.addHandler(console)
    if not debug:
        return None
    session = log_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    session.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(session / "hoopbot.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s.%(msecs)03d %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(fh)
    log.info(f"debug log and snapshots: {session}")
    return session


class Snapshots:
    """Saves the game canvas when something unexpected happens (debug mode only)."""

    MIN_GAP_S = 1.0

    def __init__(self, src, folder: Path | None):
        self.src, self.folder, self.last = src, folder, 0.0

    def __call__(self, reason: str):
        if self.folder is None or time.perf_counter() - self.last < self.MIN_GAP_S:
            return
        self.last = time.perf_counter()
        img, _ = self.src.grab((0, 0, C.REF_W, C.REF_H))
        name = f"{datetime.now():%H%M%S_%f}"[:-3] + f"_{reason}.png"
        C.write_png(self.folder / name, cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
        log.debug(f"snapshot {name}")


def _connect(args):
    if getattr(args, "replay", None):
        return ReplaySource(args.replay)
    src = ScreenSource(args.monitor)
    if src.locate() is None:
        raise SystemExit("Game canvas not found: open Swishy Hoops and keep it fully visible.")
    return src


# ---------------------------------------------------------------- watching

class Watcher:
    """Grabs frames and keeps game time, platform/hoop histories and the flying ball."""

    STATIC_FRAMES = 40

    def __init__(self, src):
        self.src = src
        self.clock = GameClock()
        self.platform = Track(PLATFORM_WINDOW_S)
        self.hoop = Track(HOOP_WINDOW_S)
        self.hoop_since = None      # game time the hoop settled at its current spot
        self.masks = deque()
        self.mask_sum = None        # running count of orange pixels over self.masks
        self.static = None
        self.scene = None
        self.ball_seen_g = None
        self.hoop_fit = None        # last fit, to tell a relocation from fast motion
        self.hoop_fit_g = -1e9
        self.max_freeze = 0.0       # longest freeze since the caller last set it to 0
        # Live: capture in a background thread so processing never waits for a grab.
        self.stream = FrameStream(src, C.FIELD, V.WORK) if isinstance(src, ScreenSource) else None
        self.seq = 0
        self.frames = 0             # processed frames, for the frame-rate report

    def grab_field(self):
        if self.stream is None:
            return self.src.grab(C.FIELD, V.WORK)
        img, t, self.seq = self.stream.next(self.seq)
        return img, t

    def tick(self):
        img, t = self.grab_field()
        last = self.scene
        hints = (last.plat if last and last.plat_score > 0.6 else None,
                 last.hoop if last and last.hoop_score > 0.7 else None)
        sc = V.read_scene(img, t, self.static, hints)
        g = self.clock.update(t, sc.gray[::8, ::8])
        sc.t = g
        self.frames += 1
        self.max_freeze = max(self.max_freeze, self.clock.frozen_for(t))
        m = sc.ball_mask.astype(np.uint8)
        if self.mask_sum is None or self.mask_sum.shape != m.shape:
            self.mask_sum, self.masks = np.zeros(m.shape, np.uint16), deque()
        self.masks.append(m)
        self.mask_sum += m
        if len(self.masks) > self.STATIC_FRAMES:
            self.mask_sum -= self.masks.popleft()
        if len(self.masks) >= 10:
            self.static = self.mask_sum > 0.6 * len(self.masks)
        if sc.plat_score > 0.6:
            self.platform.add(g, *sc.plat)
        if sc.hoop_score > 0.7:
            if self.hoop_moved(g, sc.hoop):
                log.debug(f"hoop moved to ({sc.hoop[0]:.0f}, {sc.hoop[1]:.0f})")
                self.hoop_relocates(g)
            if self.hoop_since is None:
                self.hoop_since = g
            self.hoop.add(g, *sc.hoop)
        if sc.ball is not None:
            self.ball_seen_g = g
        self.scene = sc
        return sc

    def hoop_moved(self, g, pos):
        """True if the hoop jumped to a new spot (after a score), not just moved along its path.

        Once its motion is fitted, the hoop is compared with where the fit says it should
        be. Before that, a jump only counts if it is faster than the hoop can move: after
        ~30 points it circles at ~300 px/s, i.e. 30-80 px between two slow captures."""
        last = self.hoop.last()
        if last is None:
            return False
        if self.hoop.span() >= MIN_HOOP_MOVING_S and g - self.hoop_fit_g > 0.5:
            self.hoop_fit = self.hoop.fit() or self.hoop_fit
            self.hoop_fit_g = g
        f = self.hoop_fit
        if f is not None and (f[0].moving or f[1].moving):
            return abs(f[0].at(g) - pos[0]) + abs(f[1].at(g) - pos[1]) > HOOP_RELOCATE_PX
        step = abs(last[1] - pos[0]) + abs(last[2] - pos[1])
        return step > HOOP_JUMP_PX + HOOP_MAX_SPEED * max(g - last[0], 0.0)

    def hoop_relocates(self, g):
        """The hoop jumps to a new spot after every score: start its history over."""
        self.hoop.clear()
        self.hoop_fit = None
        self.hoop_since = g

    def run_for(self, seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            self.tick()

    def score(self):
        return V.read_score(self.src.grab(C.SCORE_TEXT)[0])

    def game_over(self):
        return V.game_over(self.src.grab(C.GAME_OVER)[0])

    def game_visible(self):
        return V.logo_visible(self.src.grab(C.LOGO_BOX)[0])

    def motions(self):
        """(platform, hoop) motion fits, or a reason why not ready."""
        if self.platform.span() < MIN_PLATFORM_S:
            return None, f"watching the platform ({self.platform.span():.1f}/{MIN_PLATFORM_S} s)"
        plat = self.platform.fit()
        if plat is None:
            return None, "platform fit failed"
        if self.hoop_since is None or not self.hoop.samples:
            return None, "hoop not found"
        if self.clock.game - self.hoop_since < MIN_HOOP_STILL_S:
            return None, "hoop not settled"
        hoop = self.hoop.fit()
        if hoop is None:
            return None, "hoop fit failed"
        moving = hoop[0].moving or hoop[1].moving
        if moving and self.hoop.span() < MIN_HOOP_MOVING_S:
            return None, f"watching the moving hoop ({self.hoop.span():.1f}/{MIN_HOOP_MOVING_S} s)"
        if moving and max(hoop[0].rms, hoop[1].rms) > HOOP_FIT_RMS:
            return None, (f"hoop motion doesn't fit yet (x {hoop[0].describe()}, "
                          f"y {hoop[1].describe()})")
        return (plat, hoop), None


# ---------------------------------------------------------------- playing

class Bot:
    def __init__(self, args, src, model, snap):
        self.args, self.src, self.model, self.snap = args, src, model, snap
        self.w = Watcher(src)
        self.mouse = None
        self.n = 0
        self.trophy = False
        self.reach_logged = 0.0
        self.holds = list(args.probe_hold or [])
        self.n_probe = 0

    def next_hold(self):
        """Button hold time: a quick click, or the next probe duration (--probe-hold)."""
        if not self.holds:
            return CLICK_HOLD_S
        h = self.holds[self.n_probe % len(self.holds)]
        self.n_probe += 1
        log.info(f"probe: holding the button {h:.2f} s")
        return h

    def wait_ready(self):
        """Until no ball is flying and both motions are fitted. Returns (plat, hoop) or None."""
        start = time.perf_counter()
        last_beat = start
        while time.perf_counter() - start < IDLE_TIMEOUT_S:
            self.w.tick()
            g = self.w.clock.game
            if self.w.ball_seen_g is not None and g - self.w.ball_seen_g < SETTLE_S:
                continue
            fits, why = self.w.motions()
            if fits is not None:
                return fits
            if time.perf_counter() - last_beat > HEARTBEAT_S:
                last_beat = time.perf_counter()
                if self.w.game_over():
                    return "game over"
                sc = self.w.scene
                log.info(f"waiting: {why} (platform match {sc.plat_score:.2f}, hoop match {sc.hoop_score:.2f})")
                if sc.plat_score < 0.6 or sc.hoop_score < 0.7:
                    self.snap("not_visible")
                    if isinstance(self.src, ScreenSource) and self.src.locate() is None:
                        log.info("game window not found")
        return None

    def plan(self, fits):
        """Re-plan until shortly before the click. Returns (plan, fits) or None."""
        committed = None
        while True:
            g_now = self.w.clock.now(time.perf_counter())
            plat, hoop = fits
            horizon = 2 * np.pi / plat[1].w + 0.5 if plat[1].moving else 3.0
            g_from = g_now + (MIN_LEAD_S if committed is None else 0.02)
            p = plan_click(self.model, plat, hoop, g_from, horizon, near=committed)
            if p is None:
                log.info("no click time brings the ball down through the rim")
                return None
            if not p.exact and not (REACH_BAND[0] <= p.dx <= REACH_BAND[1]) and not self.args.probe_hold:
                # Platform and hoop can move in lockstep so that no click reaches the hoop
                # for a long time. A throw now would only cost a life: wait for it to drift.
                if time.perf_counter() - self.reach_logged > HEARTBEAT_S:
                    self.reach_logged = time.perf_counter()
                    log.info(f"hoop out of reach right now (closest rim crossing dx {p.dx:.0f}, "
                             f"need {REACH_BAND[0]}..{REACH_BAND[1]}); waiting")
                return None
            if committed is not None and abs(p.g_click - committed) > 0.15:
                log.debug(f"plan moved {p.g_click - committed:+.3f} s")
            committed = p.g_click
            wall_click = self.w.clock.to_wall(p.g_click)
            if wall_click - time.perf_counter() <= REFINE_BEFORE_S + 0.02:
                if self.w.clock.frozen_for(time.perf_counter()) > HITCH_S:
                    log.info("game hitch right before the click; waiting for the next chance")
                    self.snap("hitch")
                    self.w.run_for(0.3)
                    committed = None
                    fits, _ = self.w.motions()
                    if fits is None:
                        return None
                    continue
                return p, fits
            log.debug(f"plan: {p.describe()}, click in {(wall_click - time.perf_counter()) * 1000:.0f} ms; "
                      f"platform x {plat[0].describe()}, y {plat[1].describe()}, "
                      f"hoop x {hoop[0].describe()}, y {hoop[1].describe()}")
            end = wall_click - REFINE_BEFORE_S
            while time.perf_counter() < end:
                self.w.tick()
            new, _ = self.w.motions()
            if new is not None:
                fits = new

    def click_at(self, wall_t, hold=CLICK_HOLD_S):
        from pynput.mouse import Button
        while True:
            dt = wall_t - time.perf_counter()
            if dt <= 0:
                break
            if dt > 0.003:
                time.sleep(dt - 0.002)
        t = time.perf_counter()
        self.mouse.press(Button.left)
        time.sleep(hold)
        self.mouse.release(Button.left)
        return t

    def track_flight(self, g_click):
        """Ball and hoop positions after the click; ball [] if no ball ever showed up."""
        path, hoop = [], []
        start = time.perf_counter()
        while time.perf_counter() - start < FLIGHT_MAX_S:
            sc = self.w.tick()
            g = self.w.clock.game
            if sc.hoop_score > 0.7:
                hoop.append((g, *sc.hoop))
            if sc.ball is not None and g > g_click:
                path.append((g, *sc.ball))
            elif path and g - path[-1][0] > FLIGHT_GONE_S:
                break
            elif not path and time.perf_counter() - start > NO_FLIGHT_S:
                break
        return path, hoop

    def throw_once(self):
        ready = self.wait_ready()
        if ready == "game over":
            log.info("Game over (no lives left).")
            return False
        if ready is None:
            log.info(f"nothing to throw for {IDLE_TIMEOUT_S:.0f} s: minigame closed?")
            return False
        score_before = self.w.score()
        planned = self.plan(ready)
        if planned is None:
            self.w.run_for(0.5)
            return True
        p, (plat, hoop) = planned
        S_plat = list(self.w.platform.samples)
        S_hoop = list(self.w.hoop.samples)
        wall_click = self.w.clock.to_wall(p.g_click)
        hold = CLICK_HOLD_S
        if self.args.dry_run:
            log.info(f"[dry] would click in {(wall_click - time.perf_counter()) * 1000:.0f} ms: {p.describe()}")
            g_click = p.g_click
            while time.perf_counter() < wall_click:
                self.w.tick()
        else:
            # Never click outside a running game (game over screen, minigame closed).
            if self.w.game_over():
                log.info("Game over (no lives left).")
                return False
            if not self.w.game_visible():
                log.info("Swishy Hoops is no longer on screen; stopping without clicking.")
                self.snap("game_gone")
                return False
            hold = self.next_hold()
            t = self.click_at(wall_click, hold)
            g_click = self.w.clock.now(t)
            log.debug(f"clicked {(t - wall_click) * 1000:+.1f} ms vs plan")
        self.w.max_freeze = 0.0
        path, hoop_after = self.track_flight(g_click)
        freeze = self.w.max_freeze
        if not path and not self.args.dry_run:
            log.info("no ball flew after the click: the game ignored it; trying again")
            self.snap("click_ignored")
            return True
        self.w.run_for(SETTLE_S)
        score_after = self.w.score()
        if score_after is not None and score_before is not None and score_after > score_before:
            # It has jumped by now even if that was missed between two captures.
            self.w.hoop_relocates(self.w.clock.game)
        dx_actual = rim_crossing(path, hoop_after, S_hoop)
        t = Throw(g_click=None if self.args.dry_run else g_click, platform=S_plat, hoop=S_hoop,
                  flight=path, score_before=score_before, score_after=score_after, plan_dx=p.dx,
                  scale=self.src.canvas.scale,
                  extra={"dx_actual": dx_actual, "window": p.window, "source": "live", "hold": hold,
                         "max_freeze": freeze, "hoop_after": hoop_after})
        records.append(self.args.log, t)
        self.n += 1
        pts = t.points
        result = "miss" if pts == 0 else f"+{pts}" if pts else "?"
        log.info(f"#{self.n} {result}, score {score_after}: aimed dx {p.dx:.0f} ({p.describe()}), "
                 f"crossed at {'?' if dx_actual is None else f'{dx_actual:.0f}'}"
                 f"{'' if pts else f'  [platform x {plat[0].describe()}, y {plat[1].describe()}; hoop {hoop[0].describe()} / {hoop[1].describe()}; longest freeze {freeze:.2f} s]'}")
        if pts == 0:
            self.snap("miss")
        if not self.trophy and score_after is not None and score_after >= TROPHY_SCORE:
            self.trophy = True
            log.info(f"Score {score_after}: the {TROPHY_SCORE}+ trophy is yours. Playing on.")
        return True


def check(args):
    setup_logging(False, Path("logs"))
    src = _connect(args)
    print(f"canvas: {src.canvas}  dpr={getattr(src, 'dpr', 1.0)}")
    w = Watcher(src)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 6.0:
        w.tick()
    n = len(w.platform.samples)
    sc = w.scene
    print(f"frames: {n / (time.perf_counter() - t0):.0f} per second")
    print(f"platform: {sc.plat} (match {sc.plat_score:.2f}), hoop: {sc.hoop} (match {sc.hoop_score:.2f})")
    print(f"score: {w.score()}, game over: {w.game_over()}")
    fits, why = w.motions()
    if fits is None:
        print(f"motion: not ready ({why})")
        return
    plat, hoop = fits
    print(f"platform x {plat[0].describe()}, y {plat[1].describe()}")
    print(f"hoop x {hoop[0].describe()}, y {hoop[1].describe()}")
    if args.model.exists():
        m = Model.load(args.model)
        g = w.clock.now(time.perf_counter())
        p = plan_click(m, plat, hoop, g + MIN_LEAD_S, 2 * np.pi / plat[1].w + 0.5 if plat[1].moving else 3.0)
        print(f"plan: {p.describe() if p else 'nothing reachable'}")
        if p:
            print(f"      click in {(w.clock.to_wall(p.g_click) - time.perf_counter()) * 1000:.0f} ms")


def play(args):
    if args.replay:
        args.dry_run = True
    session = setup_logging(args.debug, Path("logs"))
    if not args.model.exists():
        raise SystemExit(f"{args.model} not found")
    model = Model.load(args.model)
    src = _connect(args)
    bot = Bot(args, src, model, Snapshots(src, session))
    if not args.dry_run:
        from pynput.mouse import Controller
        bot.mouse = Controller()
        bot.mouse.position = src.ref_to_point(*CURSOR_PARK)
        time.sleep(0.2)
    log.info(f"canvas {src.canvas}; model from {model.n_throws} throws, latency {model.latency * 1000:.0f} ms, "
             f"aiming at rim dx {model.target_dx:.0f}±{model.margin:.0f}")
    try:
        while bot.throw_once():
            if args.throws and bot.n >= args.throws:
                break
    except EOFError:
        log.info("end of recording")
    except KeyboardInterrupt:
        log.info("stopped")
