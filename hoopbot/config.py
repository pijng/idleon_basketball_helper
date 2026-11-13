"""Game geometry in reference coordinates.

Everything works in the frame of board.png (3022x1708): the game canvas at the size it
was captured. Any real window is mapped into this space by `screen.Canvas`, so model
parameters don't depend on window size.
"""
from pathlib import Path

import cv2
import numpy as np

ASSETS = Path(__file__).parent / "assets"


def read_image(path, flags=cv2.IMREAD_COLOR):
    """cv2.imread that also works for non-ASCII paths on Windows (e.g. C:\\Users\\Марк)."""
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if img is None:
        raise FileNotFoundError(path)
    return img


def write_png(path, img):
    """cv2.imwrite counterpart of read_image."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"cannot encode {path}")
    buf.tofile(str(path))


REF_W, REF_H = 3022, 1708

# Anchor templates and their top-left positions in reference space.
ANCHOR_LOGO = (2530, 36)    # "SWISHY HOOPS", assets/logo.png
ANCHOR_EXIT = (2815, 1602)  # EXIT button, assets/exit.png

# Rects are (x0, y0, x1, y1) in reference space.
# The ball's arc reaches y ~50, so the field runs to the top; static HUD (score balls,
# hearts) is removed by the static-pixel mask.
FIELD = (0, 0, 2800, REF_H)
SCORE_TEXT = (0, 70, 300, 130)       # "Score: N"; digits start at x >= SCORE_DIGITS_X
SCORE_DIGITS_X = 140
LOGO_BOX = (ANCHOR_LOGO[0] - 15, ANCHOR_LOGO[1] - 15, ANCHOR_LOGO[0] + 477, ANCHOR_LOGO[1] + 79)
GAME_OVER = (1300, 20, 1720, 110)    # "Game over!" banner, assets/gameover.png

# Template geometry. Platform template top-left = (px, py); hoop template top-left = (hx, hy).
PLATFORM_TOP = 0          # the platform's walking surface is at py
PLAYER_BOX = (-40, -420, 470, 10)    # player + ball in hand, relative to (px, py)
RIM_BOX = (-10, -10, 270, 40)        # rim band, relative to (hx, hy)
RIM_DY = 15               # ball centre crossing y = hy + RIM_DY decides the shot

BALL_AREA = (800, 16000)    # orange pixels of a ball (a full one is ~6000), reference px^2;
                            # the low end keeps a ball half off the top of the screen
