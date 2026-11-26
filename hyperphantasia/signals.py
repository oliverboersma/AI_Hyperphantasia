"""The synesthetic signal bus: what the reasoner *feels* about its own model.

Hyperphantasic people report that a mental image carries more than content:
parts are vivid or blurry, a configuration "clicks" or "feels off". These
signals are cheap, automatic and enormously useful for steering thought. Here
each one is a number computed from the imagined model, and the loop reads them
the way a person reads a gut feeling: as evidence about where to look next.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from .domains.base import ChannelView, Status


@dataclass
class Signals:
    vividness: dict[str, float] = field(default_factory=dict)  # per entity: 1 crisp .. 0 blur
    dissonance: float = 0.0       # disagreement between independent channels (0..1)
    discordant: list[str] = field(default_factory=list)  # channel names that disagree
    surprise: bool = False        # the first impression disagreed with the settled verdict
    corroboration: int = 0        # independent channels committing to the settled verdict
    abstentions: int = 0          # independent channels that could not decide
    complete: bool = True         # was the possibility space fully explored?
    perception: float = 1.0       # confidence that the scene matches the problem text
    novelty: float = 1.0          # 1 - similarity to the nearest remembered schema
    raw_confidence: float = 0.0
    confidence: float = 0.0       # after calibration against the reasoner's track record
    conflicts: list[str] = field(default_factory=list)  # premise ids implicated

    @property
    def mean_vividness(self) -> float:
        return sum(self.vividness.values()) / len(self.vividness) if self.vividness else 1.0

    def line(self) -> str:
        return (f"vividness={self.mean_vividness:.2f} dissonance={self.dissonance:.2f} "
                f"surprise={'yes' if self.surprise else 'no'} corroboration={self.corroboration} "
                f"perception={self.perception:.2f} novelty={self.novelty:.2f} "
                f"confidence={self.confidence:.2f}")


def bind(channels: list[ChannelView], settled: Status) -> tuple[float, list[str], int, int]:
    """Cross-check the independent channels.

    Returns (dissonance, discordant channel names, corroboration, abstentions).
    Dissonance is the fraction of committed channel pairs that disagree.
    """
    independent = [c for c in channels if c.independent]
    committed = [c for c in independent if c.verdict is not None]
    abstentions = len(independent) - len(committed)
    pairs = list(combinations(committed, 2))
    clashes = [(a, b) for a, b in pairs if a.verdict != b.verdict]
    dissonance = len(clashes) / len(pairs) if pairs else 0.0
    discordant = sorted({c.name for pair in clashes for c in pair})
    corroboration = sum(c.verdict == settled for c in committed)
    return dissonance, discordant, corroboration, abstentions


def raw_confidence(sig: Signals) -> float:
    """Turn internal signals into a probability-like score before calibration.

    The structure is P(scene is right) x P(reasoning is right | scene).
    Exhaustive imagination makes the second factor high; the first factor,
    perception, is where most real errors live, so it is never assumed away.
    """
    if sig.corroboration >= 2:
        reasoning = 0.99
    elif sig.corroboration == 1:
        reasoning = 0.95
    else:
        reasoning = 0.65
    if not sig.complete:
        reasoning *= 0.85
    if sig.dissonance > 0:
        reasoning *= 1 - 0.6 * sig.dissonance
    return max(0.01, min(0.995, reasoning * sig.perception))
