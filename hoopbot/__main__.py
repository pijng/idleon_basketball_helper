"""hoopbot CLI.

  hoopbot check                  # what the bot sees, without clicking
  hoopbot play [--debug]         # play until game over (keeps going after 40 points)
  hoopbot fit                    # recalibrate model.json from logged throws
"""
from __future__ import annotations

import argparse
from pathlib import Path

LIVE_LOG = Path("data") / "live_throws.jsonl"
VIDEO_LOG = Path("data") / "video_throws.jsonl"   # bundled: throws from the recordings
MODEL = Path("model.json")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="hoopbot")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("check", help="detect everything on the current screen")
    c.add_argument("--model", type=Path, default=MODEL)
    c.add_argument("--monitor", type=int)

    p = sub.add_parser("play", help="play Swishy Hoops")
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--log", type=Path, default=LIVE_LOG)
    p.add_argument("--monitor", type=int)
    p.add_argument("--dry-run", action="store_true", help="watch and plan, never click")
    p.add_argument("--replay", type=Path, help="dry-run against a recording played in real time")
    p.add_argument("--throws", type=int, default=0, help="stop after N throws (0 = until game over)")
    p.add_argument("--probe-hold", type=lambda v: [float(x) for x in v.split(",")], default=None,
                   metavar="S1,S2,...",
                   help="experiment: hold the button this long (cycling through the list) and log "
                        "the flights; throws even when the hoop looks out of reach")
    p.add_argument("--debug", action="store_true",
                   help="verbose console + logs/<session>/hoopbot.log and snapshots of odd moments")

    f = sub.add_parser("fit", help="recalibrate model.json from logged throws")
    f.add_argument("logs", type=Path, nargs="*", default=[VIDEO_LOG, LIVE_LOG])
    f.add_argument("--model", type=Path, default=MODEL)
    f.add_argument("--force", action="store_true",
                   help="save the refitted model even if it did worse on the held-out game")

    args = ap.parse_args(argv)
    if args.cmd == "check":
        from .bot import check
        check(args)
    elif args.cmd == "play":
        from .bot import play
        play(args)
    elif args.cmd == "fit":
        fit(args)


def fit(args):
    """Refit, but first check on the most recent game whether refitting helps.

    The last session is held out: the current model and a model fitted without that
    session both predict its throws. The model refitted on everything is saved only if
    the held-out one was not worse (or with --force)."""
    from dataclasses import replace
    from . import records
    from .physics import MIN_EVAL_THROWS, Model, evaluate, fit_model, sessions
    current = Model.load(args.model) if args.model.exists() else Model()
    throws = records.load(*args.logs)
    games = sessions(throws)
    test = games[-1] if games else []
    held = [t for t in throws if not any(t is u for u in test)]
    before = evaluate(current, test)
    if before is not None and before.n >= MIN_EVAL_THROWS:
        print(f"held-out check on the last game ({len(test)} throws):")
        cand = fit_model(held, replace(current), log=lambda s: None)
        after = evaluate(cand, test)
        print(f"  current model:          {before.describe()}")
        print(f"  refit without that game: {after.describe()}")
        better = after.within >= before.within - 0.02 and abs(after.bias) <= abs(before.bias) + 5
        if not better and not args.force:
            print("refitting did not help on unseen throws; model.json unchanged (--force saves anyway)")
            return
    else:
        print("the last game has too few measurable throws for a held-out check; refitting anyway")
    fit_model(throws, replace(current)).save(args.model)
    print(f"model -> {args.model}")


if __name__ == "__main__":
    main()
