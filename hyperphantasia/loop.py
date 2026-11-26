"""The Phantasia Loop: the reasoning path itself.

    ATTUNE     recall who is listening, what I have learned about my own errors,
               and whether I have seen this shape of problem before
    ENVISION   build the scene: entities, premises, the question (perception)
    GLANCE     form the first picture and read a tentative answer off it
    REFRACT    render the scene through several independent channels
    ANIMATE    run whatever dynamics the scene has
    SETTLE     reach a verdict over the whole possibility space
    BIND       cross-check the channels; dissonance means an internal fault
    CHALLENGE  confront the first impression with the counterexample search
    TRANSLATE  explain it to this listener at the confidence actually earned
    ATONE      update the self-model and the memory palace; on feedback,
               calibrate and localise the error to a stage

Every stage writes to the trace, so the whole path is inspectable afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import prod
from pathlib import Path

from .attune import (CalibrationLedger, ListenerModel, SelfModel, honesty_violations,
                     load_state, save_state)
from .backends.offline import OfflineBackend
from .domains import DOMAINS, ChannelView, Imagination, Status, Verdict
from .palace import MemoryPalace
from .scene import Scene
from .signals import Signals, bind, raw_confidence
from .translate import compose

INTUITION = {"yes": Status.NECESSARY, "no": Status.IMPOSSIBLE, "undetermined": Status.POSSIBLE,
             "contradiction": Status.INCONSISTENT, "works": Status.WORKS, "fails": Status.FAILS}


@dataclass
class Stage:
    name: str
    summary: str
    detail: str = ""


@dataclass
class Insight:
    question: str
    domain: str
    scene: Scene
    verdict: Verdict
    glance: Status | None
    signals: Signals
    channels: list[ChannelView]
    explanation: str
    trace: list[Stage] = field(default_factory=list)
    imagination: Imagination | None = None

    @property
    def status(self) -> Status:
        return self.verdict.status

    @property
    def answer(self) -> str:
        return self.verdict.summary

    @property
    def confidence(self) -> float:
        return self.signals.confidence

    def report(self, channels: bool = True) -> str:
        lines = []
        for st in self.trace:
            lines.append(f"[{st.name.upper():<9}] {st.summary}")
            if channels and st.detail:
                lines.extend("            " + ln for ln in st.detail.splitlines())
        lines.append("")
        lines.append(self.explanation)
        return "\n".join(lines)


class PhantasiaLoop:
    def __init__(self, backend=None, *, state_dir: str | Path | None = None,
                 budget: int = 200_000, glimpse: int = 2_000, imagery: str = "auto",
                 listener: ListenerModel | None = None, use_intuition: bool = False):
        self.backend = backend or OfflineBackend()
        self.state_dir = Path(state_dir) if state_dir else None
        if self.state_dir:
            self.ledger, self.me, entries = load_state(self.state_dir)
            self.palace = MemoryPalace(entries)
        else:
            self.ledger, self.me, self.palace = CalibrationLedger(), SelfModel(), MemoryPalace()
        self.budget, self.glimpse, self.imagery = budget, glimpse, imagery
        self.listener = listener or ListenerModel()
        self.use_intuition = use_intuition

    # ------------------------------------------------------------------ think
    def think(self, problem: str | Scene, listener: ListenerModel | None = None) -> Insight:
        listener = listener or self.listener
        trace: list[Stage] = []
        text = problem if isinstance(problem, str) else (problem.title or "")

        # ENVISION (perception first; attunement needs to know the shape)
        scene = problem if isinstance(problem, Scene) else self.backend.envision(problem)
        domain = DOMAINS[scene.domain]
        signature = domain.signature(scene)

        # ATTUNE
        analog, sim = self.palace.recall(signature)
        lessons = self.me.recall(scene.domain, signature)
        reliability = self.me.glance_reliability(scene.domain)
        trace.append(Stage("attune",
                           f"listener={listener.expertise}/{listener.prefers}; first-impression reliability "
                           f"in {scene.domain} = {reliability:.2f} over {self.me.runs.get(scene.domain, 0)} run(s)"
                           + (f"; analog: \"{analog['title']}\" (similarity {sim:.2f})" if analog else ""),
                           "\n".join("lesson: " + t for t in lessons)))
        trace.append(Stage("envision",
                           f"{len(scene.entities)} entities, {len(scene.relations)} premises, "
                           f"perception confidence {scene.parse_confidence:.2f}"
                           + (f", {len(scene.unparsed)} sentence(s) not understood" if scene.unparsed else ""),
                           scene.describe()))

        # Adaptive imagery: start with a glimpse, look closer only if needed.
        budget = self.budget if self.imagery == "full" else self.glimpse
        im = domain.envision(scene, budget)
        dial = "full" if self.imagery == "full" else "glimpse"
        if not im.complete and self.imagery == "auto":
            if not self._may_shortcut(domain, im, reliability, analog):
                im = domain.envision(scene, self.budget)
                dial = "glimpse -> full (picture still blurry)"

        # GLANCE
        glance = domain.glance(im)
        trace.append(Stage("glance", f"first impression: {glance.value if glance else 'none'} "
                                     f"[imagery: {dial}]",
                           domain.render_world(im, im.preferred) if im.preferred is not None else ""))

        # REFRACT
        channels = domain.refract(im)
        if self.use_intuition and hasattr(self.backend, "intuit") and text:
            channels.append(self._intuition_channel(text))
        for c in channels:
            if c.needs_complete and not im.complete and c.verdict in (Status.NECESSARY, Status.IMPOSSIBLE):
                c.verdict = None
                c.rendering += "\n(search cut short: can witness 'possible' but not prove 'necessary')"
        im.channels = channels
        trace.append(Stage("refract", "; ".join(f"{c.name}: {c.verdict.value if c.verdict else 'abstains'}"
                                                for c in channels),
                           "\n".join(f"<{c.name}> ({c.method})\n{c.rendering}" for c in channels)))

        # ANIMATE
        events = domain.animate(im)
        if events:
            trace.append(Stage("animate", "; ".join(events)))

        # SETTLE + BIND
        verdict = domain.settle(im)
        dissonance, discordant, corroboration, abstentions = bind(channels, verdict.status)
        sig = Signals(vividness=im.vividness, dissonance=dissonance, discordant=discordant,
                      corroboration=corroboration, abstentions=abstentions, complete=im.complete,
                      perception=scene.parse_confidence, novelty=1 - sim,
                      conflicts=verdict.conflict_set or sorted({x for c in channels for x in c.conflicts}))
        trace.append(Stage("bind", f"dissonance {dissonance:.2f}, {corroboration} independent channel(s) "
                                   f"corroborate, {abstentions} abstain"
                                   + (f"; DISCORD between {', '.join(discordant)}" if discordant else "")))

        # CHALLENGE
        sig.surprise = glance is not None and verdict.status not in (Status.OPEN,) and glance != verdict.status
        # The model that breaks the first impression: a counterexample if the glance said
        # "yes", a witness if it said "no".
        breaker = verdict.witness if glance is Status.IMPOSSIBLE else verdict.counterexample
        trace.append(Stage("challenge", self._challenge_summary(domain, im, verdict, glance, sig),
                           domain.render_world(im, breaker)
                           if sig.surprise and breaker is not None and domain.name != "process" else ""))

        # Perception re-check: if an independent verbal reading disagrees, look again at the scene.
        verbal = next((c for c in channels if c.name == "verbal"), None)
        if verbal and verbal.verdict and verbal.verdict != verdict.status and hasattr(self.backend, "verify"):
            faithful, issues = self.backend.verify(text, scene)
            if not faithful:
                sig.perception *= 0.5
                trace.append(Stage("challenge", "verbal reading disagreed and the scene was re-checked: "
                                                "it may misrepresent the problem", "\n".join(issues)))

        # Confidence: internal signals, then calibrated against my track record.
        sig.raw_confidence = raw_confidence(sig)
        sig.confidence = self.ledger.calibrate(scene.domain, sig.raw_confidence)
        trace.append(Stage("settle", f"{verdict.status.value}: {verdict.summary}",
                           f"signals: {sig.line()}"))

        # TRANSLATE
        explanation = compose(domain, im, verdict, sig, glance, listener, analog)
        if hasattr(self.backend, "polish"):
            polished = self.backend.polish(explanation, listener, sig.confidence)
            bad = honesty_violations(polished or "", sig.confidence)
            if polished and not bad:
                explanation = polished
            elif bad:
                trace.append(Stage("translate", "rewrite rejected: it claimed more certainty than earned ("
                                   + ", ".join(bad) + ")"))
        trace.append(Stage("translate", f"for a {listener.expertise} who prefers {listener.prefers}"))

        # ATONE (intrinsic): learn from my own surprise and discord, without waiting for a grader.
        self.me.observe(scene.domain, sig.surprise, dissonance)
        kinds = ", ".join(sorted(signature["kinds"]))
        atoned = []
        if sig.surprise:
            lesson = (f"In {scene.domain} problems built from [{kinds}], my first picture "
                      f"suggested '{glance.value}' but the answer was '{verdict.status.value}'. "
                      "Search for a second model before answering.")
            self.me.learn(scene.domain, signature, lesson)
            atoned.append("lesson stored: first impression misled")
        if dissonance > 0:
            self.me.blame("reasoning")
            self.me.learn(scene.domain, signature,
                          f"My channels {', '.join(discordant)} disagreed on a problem built from "
                          f"[{kinds}]: one of my methods is faulty there.")
            atoned.append("internal fault recorded")
        title = scene.title or (scene.relations[0].source if scene.relations else "untitled")
        self.palace.store(signature, title[:80], verdict.status.value, sig.surprise)
        atoned.append("schema filed in the memory palace")
        trace.append(Stage("atone", "; ".join(atoned)))
        self._save()

        return Insight(text, scene.domain, scene, verdict, glance, sig, channels, explanation, trace, im)

    # --------------------------------------------------------------- feedback
    def feedback(self, insight: Insight, correct: bool, note: str = "") -> str:
        """Extrinsic atonement: learn from being told whether I was right."""
        d = insight.domain
        self.ledger.record(d, insight.signals.raw_confidence, correct)
        if correct:
            msg = "Noted as correct."
        else:
            stage = self._blame(insight)
            self.me.blame(stage)
            sig = DOMAINS[d].signature(insight.scene)
            self.me.learn(d, sig, f"I was wrong on a {d} problem ({note or 'no details'}); "
                                  f"most likely fault: {stage}.")
            msg = f"Recorded the error; most likely fault: {stage}."
        acc = self.ledger.accuracy(d)
        n = len([r for r in self.ledger.records if r[0] == d])
        self._save()
        return msg + f" Track record in {d}: {acc:.0%} over {n} graded answer(s)."

    @staticmethod
    def _blame(insight: Insight) -> str:
        s = insight.signals
        if s.perception < 0.9 or insight.scene.unparsed:
            return "envision (the scene missed part of the problem)"
        if not s.complete:
            return "imagination budget (the possibility space was not fully explored)"
        if s.dissonance > 0:
            return "reasoning (independent channels disagreed)"
        # Reasoning over a complete space is exact, so the scene itself must be wrong.
        return "envision (the premises were understood differently than intended)"

    # ---------------------------------------------------------------- helpers
    def _may_shortcut(self, domain, im: Imagination, reliability: float, analog) -> bool:
        """Skip deep imagery only where I have earned trust in my first impressions."""
        if analog and analog.get("surprise"):
            return False
        if reliability < 0.95 or self.me.runs.get(im.scene.domain, 0) < 20:
            return False
        symbolic = [c for c in domain.refract(im) if c.independent and not c.needs_complete]
        return bool(symbolic) and symbolic[0].verdict is not None and symbolic[0].verdict == domain.glance(im)

    def _intuition_channel(self, text: str) -> ChannelView:
        answer, conf = self.backend.intuit(text)
        return ChannelView("verbal", f"direct verbal answer: {answer} (self-rated {conf:.2f})",
                           INTUITION.get(answer), "language model, no imagery", independent=False)

    @staticmethod
    def _challenge_summary(domain, im: Imagination, verdict: Verdict, glance, sig: Signals) -> str:
        st = verdict.status
        if st is Status.INCONSISTENT:
            return f"no model satisfies the premises; minimal clash: {', '.join(verdict.conflict_set)}"
        if domain.name == "process":
            if st is Status.FAILS:
                return "rehearsal broke the face-value reading" + \
                       (f"; a working order exists ({len(verdict.notes)} move(s))" if verdict.repair else "")
            return "rehearsal confirmed the face-value reading"
        n = prod(len(w) for w in im.worlds.values()) if isinstance(im.worlds, dict) else len(im.worlds or [])
        if sig.surprise:
            return f"counterexample found among {n} imagined model(s): the first impression does not hold"
        if st in (Status.NECESSARY, Status.IMPOSSIBLE):
            return f"searched {n} imagined model(s) for a counterexample: none exists"
        return f"{n} imagined model(s) considered"

    def _save(self) -> None:
        if self.state_dir:
            save_state(self.state_dir, self.ledger, self.me, self.palace.entries)


def reason(problem: str | Scene, **kwargs) -> Insight:
    """One-shot convenience: think about a problem with a fresh loop."""
    listener = kwargs.pop("listener", None)
    return PhantasiaLoop(**kwargs).think(problem, listener)
