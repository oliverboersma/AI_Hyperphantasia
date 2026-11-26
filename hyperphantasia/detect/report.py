"""Render evidence and verdicts as Markdown and JSON."""

from __future__ import annotations

import json

from .battery import Evidence
from .verdict import Verdict


def _f(x):
    return "n/a" if x is None else f"{x:.2f}"


def section(ev: Evidence, v: Verdict, notes: str = "") -> str:
    b = ev.behaviour
    out = [f"## {ev.subject}", "", f"**Verdict: {v.label}.** {v.text}", ""]
    out += [f"- {r}" for r in v.reasons]
    if notes:
        out += ["", notes]
    out += ["", "| behaviour | value |", "|---|---|"]
    names = {"accuracy": "accuracy (5-letter worlds)", "composition": "questions needing composition",
             "stated_pairs": "questions about stated pairs", "shuffle_consistency":
             "same answer under 4 reshuffles", "longer_worlds": "7-letter worlds (never trained)"}
    out += [f"| {names[k]} | {b[k]:.3f} |" for k in names]
    out += ["", "Linear probes for rank (chance 0.20). *scene* = after the premises, before the question.", "",
            "| layer | site | probe | untrained copy | surface counts |", "|---|---|---|---|---|"]
    for (layer, role), p in sorted(ev.probes.items()):
        out.append(f"| {layer} | {role} | **{_f(p['acc'])}** | {_f(p['untrained'])} | {_f(p['surface'])} |")
    roles = ["scene", "X", "Y", "ANS"]
    layers = sorted({k[0] for k in ev.tracing})
    out += ["", "Transplant a whole site from a world with the opposite answer: share of answers that follow it.",
            "", "| layer | " + " | ".join(roles) + " |", "|---" * (len(roles) + 1) + "|"]
    for layer in layers:
        out.append(f"| {layer} | " + " | ".join(_f(ev.tracing.get((layer, r))) for r in roles) + " |")
    if ev.subspace:
        out += ["", "Move only x (y keeps its rank). Transplant at x's site: everything, only the probe's rank "
                "subspace, or a random subspace of equal size.", "",
                "| layer | everything | rank subspace | random subspace | dims |", "|---|---|---|---|---|"]
        for layer, s in sorted(ev.subspace.items()):
            out.append(f"| {layer} | {s['full']:.2f} | **{s['rank']:.2f}** | {s['random']:.2f} | {s['dims']} |")
    return "\n".join(out)


def to_json(ev: Evidence, v: Verdict) -> dict:
    key = lambda k: f"{k[0]}:{k[1]}"
    return {"subject": ev.subject, "verdict": v.label, "reasons": v.reasons, "behaviour": ev.behaviour,
            "probes": {key(k): p for k, p in ev.probes.items()},
            "tracing": {key(k): t for k, t in ev.tracing.items()},
            "subspace": {str(k): s for k, s in ev.subspace.items()}}


def dump(path, rows) -> None:
    path.write_text(json.dumps(rows, indent=1))
