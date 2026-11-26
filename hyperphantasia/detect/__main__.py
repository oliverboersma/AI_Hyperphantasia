"""Does imagination exist in the architecture, in the learned mechanism, or neither?

    python -m hyperphantasia.detect            # calibrate, train (or load), examine, report
    python -m hyperphantasia.detect --quick    # smaller evaluation sets

1. Calibrate the detector on two hand-built subjects with known internals. If
   it does not call the scene engine "standing scene" and the surface
   heuristic "no imagination", stop: the detector itself is not trustworthy.
2. Train two networks on the yes/no answers only (nothing about the scene is
   ever supervised): a plain transformer, and one with an imagination
   bottleneck (Phantasm Slots).
3. Examine each against an untrained copy of itself, and write a report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from . import battery, report, verdict
from .subjects import NeuralSubject, SceneEngine, SurfaceHeuristic

EXPECTED = {"known positive: scene engine": "standing scene",
            "known negative: surface heuristic": "no imagination"}

NOTES = {
    "plain": "Nothing in this architecture reserves a place for a scene; whatever is found here was "
             "invented by training.",
    "attention": "The architecture forces all information about the premises through 6 slot vectors "
                 "that compete for premise tokens (Slot Attention), so transplanting the slots (the *scene* "
                 "site, layer 0) must move the answer if the network works at all. The open question is "
                 "whether the slots hold a scene (decodable ranks) or only copies of premises.",
    "anchored": "The architecture forces all information about the premises through 6 slot vectors, one "
                "placed after each premise slot, so transplanting the slots (the *scene* site, layer 0) "
                "must move the answer if the network works at all. The open question is whether the slots "
                "hold a scene (decodable ranks) or only summaries of their own premises.",
    "slots": "The architecture forces all information about the premises through 8 slot vectors, so "
             "transplanting the slots (the *scene* site, layer 0) must move the answer if the network "
             "works at all. The open question is whether what flows through the slots is a scene "
             "(decodable ranks) or some other code.",
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m hyperphantasia.detect")
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--retrain", action="store_true")
    ap.add_argument("--cache", default=".cache/detect")
    ap.add_argument("--out", default="results")
    ap.add_argument("--kinds", default="plain,slots")
    ap.add_argument("--no-curriculum", action="store_true",
                    help="mix all world sizes from the start (default: small worlds first)")
    ap.add_argument("--also", action="append", default=[],
                    help="extra trained network to examine, as PATH=NAME (e.g. an earlier run)")
    args = ap.parse_args(argv)

    sections, rows = [], []
    print("1. calibrating the detector on subjects with known internals")
    for subject in (SceneEngine(), SurfaceHeuristic()):
        ev = battery.examine(subject, quick=args.quick)
        v = verdict.judge(ev)
        ok = v.label == EXPECTED[subject.name]
        print(f"   {subject.name}: {v.label} ({'as expected' if ok else 'WRONG'})")
        if not ok:
            print("   the detector failed calibration; its verdicts cannot be trusted", file=sys.stderr)
            return 1
        sections.append(report.section(ev, v))
        rows.append(report.to_json(ev, v))

    from . import nets
    for kind in args.kinds.split(","):
        cfg = {"anchored": nets.anchored_config, "attention": nets.slot_attention_config}.get(
            kind, lambda: nets.NetConfig(kind=kind))()
        cur = not args.no_curriculum
        path = Path(args.cache) / f"{kind}_seed{args.seed}_steps{args.steps}{'_curriculum' if cur else ''}.pkl"
        if path.exists() and not args.retrain:
            cfg, params, log = nets.load(path)
            print(f"2. loaded trained {kind} network from {path}")
        else:
            print(f"2. training {kind} network (answers only, {args.steps} steps max)")
            t = time.time()
            params, log = nets.train(cfg, steps=args.steps, seed=args.seed, curriculum=cur,
                                     checkpoint=path.with_suffix(".ckpt"))
            nets.save(path, cfg, params, log)
            print(f"   trained in {time.time() - t:.0f}s")
        name = {"plain": "plain transformer", "slots": "Phantasm Slots transformer",
                "anchored": "Phantasm Slots transformer (anchored slots)",
                "attention": "Phantasm Slots transformer (slot attention)"}[kind]
        subject = NeuralSubject(cfg, params, name)
        control = NeuralSubject(cfg, nets.init(cfg, args.seed + 99), f"untrained {name}")
        print(f"3. examining {name}")
        ev = battery.examine(subject, control, quick=args.quick)
        v = verdict.judge(ev)
        print(f"   verdict: {v.label}")
        for r in v.reasons:
            print(f"     - {r}")
        sections.append(report.section(ev, v, NOTES[kind]))
        rows.append(report.to_json(ev, v))

    for spec in args.also:
        path, name = spec.split("=", 1)
        cfg, params, _ = nets.load(Path(path))
        subject = NeuralSubject(cfg, params, name)
        control = NeuralSubject(cfg, nets.init(cfg, args.seed + 99), f"untrained {name}")
        print(f"3. examining {name}")
        ev = battery.examine(subject, control, quick=args.quick)
        v = verdict.judge(ev)
        print(f"   verdict: {v.label}")
        sections.append(report.section(ev, v))
        rows.append(report.to_json(ev, v))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = ["# Imagination detector report", "",
              "Verdict per subject, from behaviour, linear probes, and interchange interventions. "
              "See docs/IMAGINATION_DETECTOR.md for the method.", "",
              "| subject | verdict |", "|---|---|"]
    header += [f"| {r['subject']} | {r['verdict']} |" for r in rows]
    (out / "imagination_report.md").write_text("\n".join(header) + "\n\n" + "\n\n".join(sections) + "\n")
    report.dump(out / "imagination_report.json", rows)
    print(f"report written to {out / 'imagination_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
