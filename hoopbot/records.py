"""Throw records: what `fit` learns from (JSON lines)."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Throw:
    g_click: float | None                     # game time of the click
    platform: list[tuple[float, float, float]]  # (g, x, y) before the click
    hoop: list[tuple[float, float, float]]      # (g, x, y) before the click
    flight: list[tuple[float, float, float]]    # (g, x, y) of the ball after the click
    score_before: int | None = None
    score_after: int | None = None
    plan_dx: float | None = None
    scale: float | None = None
    extra: dict = field(default_factory=dict)

    @property
    def points(self):
        if self.score_before is None or self.score_after is None:
            return None
        return self.score_after - self.score_before


def append(path: Path, t: Throw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(asdict(t)) + "\n")


def load(*paths: Path) -> list[Throw]:
    out = []
    for p in paths:
        if Path(p).exists():
            out += [Throw(**json.loads(line)) for line in open(p) if line.strip()]
    return out
