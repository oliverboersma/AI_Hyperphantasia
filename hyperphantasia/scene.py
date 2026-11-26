"""The Scene: what the reasoner has *imagined* from a problem statement.

A Scene is the binding registry of the architecture. Every entity and every
premise gets a stable id, and every channel (picture, graph, diagram,
simulation) refers back to those ids. Because the ids are shared, a conflict
found in one channel can be traced to the exact premise that caused it, and an
explanation can cite its sources. This is the software version of synesthetic
binding: one concept, many renderings, one identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


def slug(text: str) -> str:
    """Normalise a surface phrase ("the Red cup") to an entity id ("red_cup")."""
    text = text.strip().lower()
    text = re.sub(r"^(the|a|an)\s+", "", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return text or "thing"


def display(text: str) -> str:
    """Normalise a surface phrase for display ("The red cup" -> "red cup")."""
    text = text.strip()
    text = re.sub(r"^(the|a|an)\s+", "", text, flags=re.IGNORECASE)
    return text.strip()


def singular(phrase: str) -> str:
    words = phrase.strip().split()
    if not words:
        return phrase
    w = words[-1]
    if w.lower().endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif re.search(r"(sses|shes|ches|xes)$", w.lower()):
        w = w[:-2]
    elif w.lower().endswith("s") and not w.lower().endswith("ss") and len(w) > 3:
        w = w[:-1]
    return " ".join(words[:-1] + [w])


def plural(phrase: str) -> str:
    words = phrase.split()
    if not words:
        return phrase
    w = words[-1]
    if re.search(r"[^aeiou]y$", w):
        w = w[:-1] + "ies"
    elif re.search(r"(s|sh|ch|x|z)$", w):
        w = w + "es"
    else:
        w = w + "s"
    return " ".join(words[:-1] + [w])


def with_article(noun: str) -> str:
    return ("an " if noun[:1].lower() in "aeiou" else "a ") + noun


@dataclass(frozen=True)
class Relation:
    """One premise, in canonical form, with provenance.

    kind   -- domain vocabulary word, e.g. "left_of", "all", "requires"
    args   -- entity ids
    rid    -- binding id ("p1", "p2", ...) used to cite and localise conflicts
    source -- the sentence that produced it
    """

    kind: str
    args: tuple[str, ...]
    rid: str = ""
    source: str = ""


@dataclass(frozen=True)
class Query:
    kind: str
    args: tuple[str, ...]
    text: str = ""


@dataclass
class Scene:
    domain: str
    entities: dict[str, str] = field(default_factory=dict)  # id -> label
    relations: list[Relation] = field(default_factory=list)
    query: Query | None = None
    title: str = ""
    extras: dict[str, Any] = field(default_factory=dict)
    # Perception bookkeeping: sentences we could not turn into structure.
    # Anything here is a hole in the imagined scene and lowers confidence.
    unparsed: list[str] = field(default_factory=list)
    parse_confidence: float = 1.0

    def entity(self, phrase: str) -> str:
        """Register (or look up) an entity by surface phrase; return its id."""
        eid = slug(phrase)
        self.entities.setdefault(eid, display(phrase))
        return eid

    def add(self, kind: str, *args: str, source: str = "") -> Relation:
        rel = Relation(kind, tuple(args), f"p{len(self.relations) + 1}", source.strip())
        self.relations.append(rel)
        return rel

    def label(self, eid: str) -> str:
        return self.entities.get(eid, eid)

    def premise(self, rid: str) -> Relation | None:
        return next((r for r in self.relations if r.rid == rid), None)

    def describe(self) -> str:
        lines = [f"domain: {self.domain}"]
        if self.title:
            lines.append(f"title: {self.title}")
        lines.append("entities: " + ", ".join(f"{k} ({v})" for k, v in self.entities.items()))
        for r in self.relations:
            lines.append(f"  [{r.rid}] {r.kind}({', '.join(r.args)})  <- {r.source!r}")
        if self.query:
            lines.append(f"query: {self.query.kind}({', '.join(self.query.args)})")
        if self.unparsed:
            lines.append("unparsed: " + " | ".join(self.unparsed))
        return "\n".join(lines)
