"""Translate: turn the imagined model into an explanation for one listener.

A good explanation of a mental model is a guided tour of it, not a dump of the
reasoning trace: show the picture, state the answer at the confidence that
was earned, point to the premises that force it, and, when the obvious
picture misleads, show the second picture that breaks it. The order and
density of that tour adapt to the listener.
"""

from __future__ import annotations

from .attune import ListenerModel, hedge
from .domains.base import Domain, Imagination, Status, Verdict
from .signals import Signals


def _indent(block: str, pad: str = "    ") -> str:
    return "\n".join(pad + line for line in block.splitlines())


def _world_text(domain: Domain, im: Imagination, world, listener: ListenerModel) -> str:
    if listener.prefers == "words" and hasattr(domain, "verbal"):
        return _indent(domain.verbal(im, world))
    return _indent(domain.render_world(im, world))


def compose(domain: Domain, im: Imagination, verdict: Verdict, sig: Signals,
            glance: Status | None, listener: ListenerModel, analog: dict | None) -> str:
    out: list[str] = [verdict.summary, f"{hedge(sig.confidence)} (confidence {sig.confidence:.0%})."]
    expert = listener.expertise == "expert"
    st = verdict.status

    if st is Status.INCONSISTENT:
        out.append("No single picture can hold all of these at once; drop any one of the "
                   f"clashing premises ({', '.join(verdict.conflict_set)}) and a picture forms.")
        return "\n\n".join(out)

    if domain.name == "process":
        if verdict.repair and listener.prefers != "words" and not expert:
            out.append("Rehearse it like this, one room per step:")
            palace = next((c for c in im.channels if c.name == "palace"), None)
            if palace:
                out.append(palace.rendering)
        elif verdict.repair:
            out.append("Working order: " + domain.render_world(im, verdict.repair))
        elif st is Status.WORKS and not expert:
            timeline = next((c for c in im.channels if c.name == "timeline"), None)
            if timeline:
                out.append("How the steps depend on each other:\n" + timeline.rendering)
        if glance is Status.WORKS and st is Status.FAILS and not expert:
            out.append("Read at face value the steps look fine; it is only when they are rehearsed "
                       "in order that the broken dependency shows up.")
        return "\n\n".join(out)

    if st is Status.POSSIBLE and verdict.counterexample is not None:
        if sig.surprise and not expert:
            out.append("Careful: the first picture that comes to mind makes it look settled. That "
                       "picture fits the premises, but it is not the only one.")
        out.append("An arrangement that fits every premise, where it holds:\n"
                   + _world_text(domain, im, verdict.witness, listener))
        out.append("Another that fits every premise, where it does not:\n"
                   + _world_text(domain, im, verdict.counterexample, listener))
    elif st in (Status.NECESSARY, Status.IMPOSSIBLE, Status.OPEN) and im.preferred is not None:
        if not expert or listener.prefers == "picture":
            out.append("How I picture it:\n" + _world_text(domain, im, im.preferred, listener))

    proofs = [p for p in im.data.get("proofs", []) if p]
    if proofs and st in (Status.NECESSARY, Status.IMPOSSIBLE):
        out.append("Why: " + proofs[0])
    if sig.corroboration >= 2:
        names = [c.name for c in im.channels if c.independent and c.verdict == st]
        out.append(f"Checked by {sig.corroboration} independent methods ({', '.join(names)}); they agree.")
    if sig.dissonance > 0:
        out.append("Warning: my internal methods disagree here (" + ", ".join(sig.discordant) +
                   "), which means one of them is faulty. Treat this answer with suspicion.")
    if analog and analog.get("surprise") and not expert:
        out.append(f"This has the same shape as an earlier problem (\"{analog['title']}\") where the "
                   "first picture was also misleading.")
    return "\n\n".join(out)
