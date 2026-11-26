"""Shared vocabulary for every imaginative domain.

A *domain* knows how to turn a Scene into a space of concrete possibilities
(worlds), how to render that space through several independent channels, and
how to search it for counterexamples. The loop in ``hyperphantasia.loop`` is
domain-agnostic; it only talks to the interface defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..scene import Scene


class Status(str, Enum):
    NECESSARY = "necessary"        # true in every world consistent with the premises
    IMPOSSIBLE = "impossible"      # false in every such world
    POSSIBLE = "possible"          # true in some, false in others
    INCONSISTENT = "inconsistent"  # no world satisfies the premises
    WORKS = "works"                # process: the procedure reaches its goal
    FAILS = "fails"                # process: it does not
    OPEN = "open"                  # nothing was asked; the scene was modelled


DECISIVE = {Status.NECESSARY, Status.IMPOSSIBLE, Status.WORKS, Status.FAILS}


@dataclass
class ChannelView:
    """One rendering of the imagined scene ("one sense").

    ``verdict`` is the answer this channel reaches *by its own method*, or
    None when the channel abstains (cannot decide). Disagreement between two
    channels that both commit to an answer is dissonance.
    """

    name: str
    rendering: str
    verdict: Status | None = None
    method: str = ""
    conflicts: list[str] = field(default_factory=list)  # premise ids implicated
    independent: bool = True  # False for channels that are pure projections of another
    # True for exhaustive-search channels: if the search was cut short they can
    # still prove "possible" (both cases witnessed) but not "necessary".
    needs_complete: bool = False


@dataclass
class Verdict:
    status: Status
    summary: str
    witness: Any = None          # a world where the claim holds
    counterexample: Any = None   # a world where it fails
    conflict_set: list[str] = field(default_factory=list)  # minimal clashing premises
    repair: Any = None           # process: a corrected ordering
    notes: list[str] = field(default_factory=list)


@dataclass
class Imagination:
    """The full imagined state: the scene plus everything built from it."""

    scene: Scene
    worlds: Any = None                  # domain-specific possibility space
    preferred: Any = None               # the first world that comes to mind
    complete: bool = True               # False if enumeration hit the budget
    vividness: dict[str, float] = field(default_factory=dict)
    channels: list[ChannelView] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)  # domain scratch space


class Domain:
    """Interface every domain implements. See spatial.py for the reference."""

    name = "base"

    def envision(self, scene: Scene, budget: int) -> Imagination:
        raise NotImplementedError

    def glance(self, im: Imagination) -> Status | None:
        """System-1: what the single preferred picture suggests."""
        raise NotImplementedError

    def refract(self, im: Imagination) -> list[ChannelView]:
        raise NotImplementedError

    def animate(self, im: Imagination) -> list[str]:
        """Run any dynamics the scene has. Default: a static scene."""
        return []

    def settle(self, im: Imagination) -> Verdict:
        raise NotImplementedError

    def render_world(self, im: Imagination, world: Any) -> str:
        raise NotImplementedError

    def signature(self, scene: Scene) -> dict[str, Any]:
        """Structural fingerprint used by the memory palace for analogy."""
        kinds: dict[str, int] = {}
        for r in scene.relations:
            kinds[r.kind] = kinds.get(r.kind, 0) + 1
        return {
            "domain": scene.domain,
            "kinds": kinds,
            "n_entities": len(scene.entities),
            "query": scene.query.kind if scene.query else None,
        }
