"""The memory palace: a library of mental-model *shapes*.

After a problem is solved, its structure (not its content) is stored: which
kinds of relations, how many entities, what was asked, whether the first
impression misled. A new problem is compared by shape, so the reasoner can say
"this is the same shape as the cutlery problem, where my first picture fooled
me", which is analogy (Gentner's structure mapping) in its simplest
operational form. It is also how one person's hard-won lesson becomes a
reusable schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def similarity(a: dict, b: dict) -> float:
    if a.get("domain") != b.get("domain"):
        return 0.0
    ka, kb = a.get("kinds", {}), b.get("kinds", {})
    keys = set(ka) | set(kb)
    shared = sum(min(ka.get(k, 0), kb.get(k, 0)) for k in keys)
    total = sum(max(ka.get(k, 0), kb.get(k, 0)) for k in keys) or 1
    na, nb = a.get("n_entities", 0), b.get("n_entities", 0)
    size = 1 - abs(na - nb) / max(na, nb, 1)
    same_q = 1.0 if a.get("query") == b.get("query") else 0.0
    return 0.7 * shared / total + 0.15 * size + 0.15 * same_q


@dataclass
class MemoryPalace:
    entries: list[dict] = field(default_factory=list)
    threshold: float = 0.75

    def recall(self, signature: dict) -> tuple[dict | None, float]:
        best, score = None, 0.0
        for e in self.entries:
            s = similarity(signature, e["signature"])
            if s > score:
                best, score = e, s
        return (best, score) if score >= self.threshold else (None, score)

    def store(self, signature: dict, title: str, status: str, surprise: bool) -> None:
        for e in self.entries:
            if e["signature"] == signature and e["title"] == title:
                e["uses"] += 1
                e["surprise"] = e["surprise"] or surprise
                return
        self.entries.append({"signature": signature, "title": title, "status": status,
                             "surprise": surprise, "uses": 1})
