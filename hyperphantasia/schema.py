"""Validation between free-form perception and the structured scene.

Whatever builds a Scene (the offline parser, an LLM, a person writing JSON),
its output passes through here. Unknown relation kinds, wrong arities and
dangling references are not silently dropped: they become ``unparsed`` holes
that lower perception confidence, so the reasoner knows what it failed to see.
"""

from __future__ import annotations

from typing import Any

from .domains.categorical import KINDS as CATEGORICAL_KINDS
from .domains.process import KINDS as PROCESS_KINDS
from .domains.spatial import EXPANSION
from .scene import Query, Scene, display, slug

ARITY: dict[str, dict[str, int]] = {
    "spatial": {k: (3 if k == "between" else 2) for k in EXPANSION},
    "categorical": {k: 2 for k in CATEGORICAL_KINDS},
    "process": {k: 2 for k in PROCESS_KINDS},
}
QUERY_ARITY: dict[str, dict[str, int]] = {
    "spatial": {**ARITY["spatial"], "relative": 2},
    "categorical": ARITY["categorical"],
    "process": {"achieves": 0},
}
ALL_RELATION_KINDS = sorted({k for d in ARITY.values() for k in d})
ALL_QUERY_KINDS = sorted({k for d in QUERY_ARITY.values() for k in d} | {"none"})


def perception_confidence(base: float, n_unparsed: int) -> float:
    """Each sentence we could not represent is a potential missing premise."""
    return max(0.2, min(1.0, base) * (1 - 0.3 * n_unparsed))


def scene_from_dict(data: dict[str, Any]) -> Scene:
    domain = data.get("domain")
    if domain not in ARITY:
        raise ValueError(f"unknown domain {domain!r}")
    scene = Scene(domain=domain, title=data.get("title", ""))
    labels = {slug(e["id"]): display(e.get("label") or e["id"]) for e in data.get("entities", [])}
    for eid, label in labels.items():
        scene.entities[eid] = label

    def ref(x: str) -> str | None:
        eid = slug(x)
        if eid not in scene.entities:
            if domain == "process":  # facts may be introduced implicitly
                scene.entities[eid] = display(x.replace("_", " "))
                return eid
            return None
        return eid

    unparsed = list(data.get("unparsed", []))
    for r in data.get("relations", []):
        kind, args = r.get("kind"), r.get("args", [])
        src = r.get("source", "")
        if ARITY[domain].get(kind) != len(args):
            unparsed.append(f"{src or kind} (unsupported relation {kind}/{len(args)})")
            continue
        ids = [ref(a) for a in args]
        if None in ids:
            unparsed.append(f"{src or kind} (unknown entity)")
            continue
        scene.add(kind, *ids, source=src or f"{kind}({', '.join(args)})")

    q = data.get("query")
    if q and q.get("kind") not in (None, "none"):
        kind, args = q["kind"], q.get("args", [])
        if QUERY_ARITY[domain].get(kind) == len(args):
            ids = [ref(a) for a in args]
            if None not in ids:
                scene.query = Query(kind, tuple(ids), q.get("text", ""))
            else:
                unparsed.append(f"question: {q.get('text', kind)} (unknown entity)")
        else:
            unparsed.append(f"question: {q.get('text', kind)} (unsupported)")
    if domain == "process":
        scene.query = Query("achieves", (), "Does the procedure reach its goal?")
        scene.extras["steps"] = [s for s in (ref(x) for x in data.get("steps", [])) if s]
        scene.extras["initial"] = [s for s in (ref(x) for x in data.get("initial", [])) if s]
        scene.extras["goal"] = [s for s in (ref(x) for x in data.get("goal", [])) if s]
    if data.get("row"):
        scene.extras["row"] = True
    if domain == "categorical":
        scene.extras["existential_import"] = data.get("existential_import", True)
    scene.unparsed = unparsed
    scene.parse_confidence = perception_confidence(float(data.get("parse_confidence", 1.0)), len(unparsed))
    return scene
