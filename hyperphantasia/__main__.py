"""Command line: python -m hyperphantasia "problem text" [options]."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .attune import ListenerModel
from .loop import PhantasiaLoop
from .schema import scene_from_dict


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="hyperphantasia", description=__doc__)
    ap.add_argument("problem", nargs="?", help="problem text, a path to a .txt/.json file, or '-' for stdin")
    ap.add_argument("--claude", action="store_true", help="use Claude for perception and voice")
    ap.add_argument("--intuition", action="store_true", help="with --claude: add a verbal first-impression channel")
    ap.add_argument("--listener", default="", help="e.g. 'expert words' or 'novice picture' or 'steps'")
    ap.add_argument("--trace", action="store_true", help="print the full reasoning path")
    ap.add_argument("--state", help="directory for persistent self-model, calibration and memory palace")
    ap.add_argument("--imagery", choices=["auto", "full"], default="auto")
    ap.add_argument("--json", action="store_true", help="print a machine-readable summary")
    args = ap.parse_args(argv)

    raw = sys.stdin.read() if args.problem in (None, "-") else args.problem
    try:
        if "\n" not in raw and Path(raw).is_file():
            raw = Path(raw).read_text()
    except OSError:  # e.g. a long one-line problem is not a usable file name
        pass
    problem = scene_from_dict(json.loads(raw)) if raw.lstrip().startswith("{") else raw

    backend = None
    if args.claude:
        from .backends.claude import ClaudeBackend
        backend = ClaudeBackend()
    loop = PhantasiaLoop(backend, state_dir=args.state, imagery=args.imagery,
                         listener=ListenerModel.parse(args.listener), use_intuition=args.intuition)
    insight = loop.think(problem)
    if args.json:
        print(json.dumps({
            "domain": insight.domain, "status": insight.status.value, "answer": insight.answer,
            "confidence": round(insight.confidence, 4), "first_impression":
                insight.glance.value if insight.glance else None,
            "surprise": insight.signals.surprise, "dissonance": insight.signals.dissonance,
            "conflicts": insight.signals.conflicts,
            "channels": {c.name: (c.verdict.value if c.verdict else None) for c in insight.channels},
        }, indent=2))
    else:
        print(insight.report() if args.trace else insight.explanation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
