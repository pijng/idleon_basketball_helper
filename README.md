# Idleon Hoops Helper

**A bot for the Swishy Hoops minigame in Legends of Idleon. Goal: the "score 40+" trophy.**

It watches the screen, predicts the whole flight of the ball from the moving platform and
the hoop, and clicks at the moment that drops the ball through the rim. After 40 points it
says so in the log and keeps playing until game over.

- It only looks at the screen and clicks the mouse: no game files, no memory reading, no injection.
- Any window size and any display scaling.
- Handles the game's hitches and clicks the game ignores (details below).

> Status: the 40+ trophy has been won with it (score 42 in that game). The model is fitted
> on 87 live throws plus 10 from screen recordings.

## Install (Windows)

1. Install **Python 3.12** from [python.org](https://www.python.org/downloads/windows/)
   and tick **"Add python.exe to PATH"**.
2. Download this project (**Code → Download ZIP**, unzip) or `git clone` it.
3. In **PowerShell**, in the project folder:

   ```powershell
   py -3.12 -m venv .venv
   .venv\Scripts\activate
   pip install -e .
   ```

macOS: the same with `python3 -m venv .venv` and `.venv/bin/activate`. Give your terminal
**Screen Recording** and **Accessibility** permission.

## Quick start

1. Open **Swishy Hoops**. Keep the whole game visible (including the "SWISHY HOOPS" title
   and the EXIT button) and not covered by the terminal. A larger window is more precise.
2. Run:

   ```powershell
   hoopbot play
   ```

3. Don't touch the mouse. The bot watches the platform for ~3 s, then starts throwing.
   It stops at game over; **Ctrl+C** stops it earlier.

Each throw prints a line like
`#12 +2, score 18: aimed dx 151 (rim crossing dx 151, window 236 ms), crossed at 149`.
`Score 40: the 40+ trophy is yours. Playing on.` marks the trophy.

`hoopbot check` prints what the bot sees (window, platform, hoop, score, their motion) and
when it would click, without clicking.

## Features

| command | what it does |
| --- | --- |
| `hoopbot play` | plays until game over, also after 40 points |
| `hoopbot play --debug` | step-by-step log, plus `logs/<date-time>/` with a log file and screenshots of misses and odd moments |
| `hoopbot play --throws 20` | stops after 20 throws |
| `hoopbot play --dry-run` | watches and plans, never clicks |
| `hoopbot play --replay video.mov` | dry-run against a screen recording, for testing |
| `hoopbot play --monitor N` | capture only monitor N |
| `hoopbot check` | one-shot detection report |
| `hoopbot fit` | recalibrates `model.json` from the throws in `data/live_throws.jsonl` |

What the bot handles on its own:

- **Moving platform** (up and down, later in a circle): measured continuously, fitted
  as a sine per axis.
- **Hoop jumps after a throw, and moves later on**: the bot waits until the hoop has
  settled, or has been watched long enough to fit its motion. It then aims for where the
  hoop will be when the ball arrives.
- **Game hitches**: the picture freezes for up to ~0.6 s. The game clock doesn't simply
  stop or keep running; a freeze advances it by about 0.25 s at most. The bot keeps its
  own game clock with the same rule, so its motion fits stay right across hitches. If the
  game hitches right before a planned click, it waits for the next chance.
- **Ignored clicks**: if no ball flies after a click, it throws again.
- **Game over**: detected from the "Game over!" banner.

## How it works

- **Finding the game.** It matches the title and the EXIT button anywhere on screen. All
  coordinates are converted into the 3022x1708 frame of `board.png`.
- **Vision**, at half resolution, ~20 frames per second on a MacBook: the screen is
  captured in a background thread, and platform and hoop are searched near where they
  were a frame ago (the whole field only if that fails).
  - platform and hoop by template matching (the hoop by its rim, so a hoop low enough to
    have its net cut off by the screen edge is still found);
  - the flying ball as an orange blob (minus the player, the rim and the HUD);
  - the score by glyph matching.
- **The shot.** The player jumps by itself after the click; the ball leaves the hand
  ~0.55 s after the click. The ball inherits part of the platform's motion, both its
  position and its velocity, per axis. The platform's state is taken ~0.35 s after the
  click. Then the ball flies a plain parabola: horizontal speed ~1235 px/s, gravity
  ~2250 px/s². So the moment of the click decides the whole flight. All of these
  constants are fitted on whole tracked flight paths; the median path error is ~16 px.
- **Aiming.** For every possible click time over the next platform cycle (4 ms steps),
  it predicts where the ball comes down through the rim level. It picks the middle of the
  longest stretch of click times that land within ±35 px of the aim point. Every logged
  throw that crossed the rim at 100–175 px from the hoop's left edge scored; the aim point is
  the median of logged swishes (~135). If no click time reaches that window (platform and hoop
  moving in lockstep), it still throws when the ball would land at 95–225 px (long shots
  there usually bank in off the backboard); otherwise it waits for the hoop to drift.
- **Learning.** Every throw is logged with the click time, the platform and hoop
  histories, the ball's flight and the score. `hoopbot fit` recalibrates the click
  latency, the release point, the launch speeds and the aim point from them.

## Calibration

The bundled `model.json` comes from `data/video_throws.jsonl` (recordings) and the first
live sessions. Every live throw is appended to `data/live_throws.jsonl`; after a game or
two on a new machine (at least 10 throws), run:

```powershell
hoopbot fit
```

It refits the click-to-release delay, what the ball inherits from the platform, the
launch speeds and gravity, using both files. Throws with a long game hitch are skipped.
With 10 or more clean swishes logged, it also moves the aim point to their median.

Before saving, `fit` checks itself on the most recent game: it refits without that game
and compares how well the current and the refitted model predict where those balls really
crossed the rim. It prints both and leaves `model.json` alone if refitting didn't help
(`--force` saves anyway).

## License

MIT, see [LICENSE](LICENSE).
