"""Self-attunement: the reasoner modelling itself, its listener, and its errors.

Three small models, all persistent:

* CalibrationLedger -- how often am I right when I feel this confident?
                       Maps raw internal confidence to calibrated probability.
* SelfModel         -- where do my first impressions fail? What went wrong
                       before, and in which stage? (atonement: owning an error
                       means localising it and changing future behaviour.)
* ListenerModel     -- who am I explaining this to, and how do they think?

Plus the honesty invariant: the words used to express certainty must match
the calibrated confidence. It is checked mechanically, not hoped for.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path


# --------------------------------------------------------------- calibration
@dataclass
class CalibrationLedger:
    records: list[tuple[str, float, bool]] = field(default_factory=list)
    min_records: int = 5
    prior_strength: float = 4.0  # pseudo-observations at the raw value

    def record(self, domain: str, confidence: float, correct: bool) -> None:
        self.records.append((domain, float(confidence), bool(correct)))

    def calibrate(self, domain: str, raw: float) -> float:
        """Histogram-binned calibration, shrunk toward the raw score."""
        hist = [(c, ok) for d, c, ok in self.records if d == domain]
        if len(hist) < self.min_records:
            return raw
        lo, hi = max(0.0, raw - 0.1), min(1.0, raw + 0.1)
        near = [ok for c, ok in hist if lo <= c <= hi]
        hits = sum(near)
        k = self.prior_strength
        return (hits + k * raw) / (len(near) + k)

    def brier(self, domain: str | None = None) -> float | None:
        rows = [(c, ok) for d, c, ok in self.records if domain in (None, d)]
        if not rows:
            return None
        return sum((c - ok) ** 2 for c, ok in rows) / len(rows)

    def accuracy(self, domain: str | None = None) -> float | None:
        rows = [ok for d, _, ok in self.records if domain in (None, d)]
        return sum(rows) / len(rows) if rows else None


# ---------------------------------------------------------------- self model
@dataclass
class SelfModel:
    runs: dict[str, int] = field(default_factory=dict)
    glance_misses: dict[str, int] = field(default_factory=dict)
    dissonance_events: dict[str, int] = field(default_factory=dict)
    blamed: dict[str, int] = field(default_factory=dict)  # stage -> count
    lessons: list[dict] = field(default_factory=list)

    def observe(self, domain: str, surprise: bool, dissonance: float) -> None:
        self.runs[domain] = self.runs.get(domain, 0) + 1
        if surprise:
            self.glance_misses[domain] = self.glance_misses.get(domain, 0) + 1
        if dissonance > 0:
            self.dissonance_events[domain] = self.dissonance_events.get(domain, 0) + 1

    def glance_reliability(self, domain: str) -> float:
        """Laplace-smoothed rate at which my first picture is already right."""
        n = self.runs.get(domain, 0)
        miss = self.glance_misses.get(domain, 0)
        return (n - miss + 1) / (n + 2)

    def learn(self, domain: str, signature: dict, text: str) -> None:
        for lesson in self.lessons:
            if lesson["text"] == text:
                lesson["count"] += 1
                return
        self.lessons.append({"domain": domain, "kinds": sorted(signature.get("kinds", {})),
                             "text": text, "count": 1})

    def recall(self, domain: str, signature: dict, limit: int = 3) -> list[str]:
        kinds = set(signature.get("kinds", {}))
        scored = []
        for lesson in self.lessons:
            if lesson["domain"] != domain:
                continue
            overlap = len(kinds & set(lesson["kinds"])) / max(1, len(kinds | set(lesson["kinds"])))
            scored.append((overlap * lesson["count"], lesson["text"]))
        return [t for s, t in sorted(scored, reverse=True)[:limit] if s > 0]

    def blame(self, stage: str) -> None:
        self.blamed[stage] = self.blamed.get(stage, 0) + 1


# ------------------------------------------------------------ listener model
@dataclass
class ListenerModel:
    prefers: str = "picture"   # "picture" | "words" | "steps"
    expertise: str = "novice"  # "novice" | "expert"

    @classmethod
    def parse(cls, spec: str | None) -> "ListenerModel":
        lm = cls()
        for part in (spec or "").replace(",", " ").split():
            if part in ("picture", "words", "steps"):
                lm.prefers = part
            elif part in ("novice", "expert"):
                lm.expertise = part
        return lm


# ---------------------------------------------------------------- honesty
BANDS = [
    (0.97, "I'm certain of this"),
    (0.90, "I'm highly confident"),
    (0.75, "I'm fairly confident"),
    (0.55, "I lean this way, but I'm not sure"),
    (0.00, "I don't really know"),
]
OVERCLAIM = re.compile(r"\b(certain(ly)?|definitely|guaranteed|without (a )?doubt|no doubt|"
                       r"undeniabl[ey]|100 ?%|absolutely sure)\b", re.IGNORECASE)


def hedge(confidence: float) -> str:
    return next(text for floor, text in BANDS if confidence >= floor)


def honesty_violations(text: str, confidence: float) -> list[str]:
    """Certainty words a speaker at this confidence has not earned."""
    if confidence >= BANDS[0][0]:
        return []
    return sorted({m.group(0).lower() for m in OVERCLAIM.finditer(text)})


# ------------------------------------------------------------- persistence
def save_state(path: Path, ledger: CalibrationLedger, me: SelfModel, palace_entries: list) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "ledger.json").write_text(json.dumps(ledger.records))
    (path / "self.json").write_text(json.dumps(asdict(me), indent=1))
    (path / "palace.json").write_text(json.dumps(palace_entries, indent=1))


def load_state(path: Path) -> tuple[CalibrationLedger, SelfModel, list]:
    ledger, me, palace = CalibrationLedger(), SelfModel(), []
    if (path / "ledger.json").exists():
        ledger.records = [tuple(r) for r in json.loads((path / "ledger.json").read_text())]
    if (path / "self.json").exists():
        me = SelfModel(**json.loads((path / "self.json").read_text()))
    if (path / "palace.json").exists():
        palace = json.loads((path / "palace.json").read_text())
    return ledger, me, palace
