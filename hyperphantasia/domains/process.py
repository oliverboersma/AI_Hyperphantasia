"""Procedural imagination: rehearsing a process in the mind's eye.

A process is a list of steps. Each step needs some facts to be true, makes
some facts true, and ends some facts. Memorising a process well means being
able to *walk through it*: to feel which step depends on which, and where a
step quietly destroys the condition a later step needs.

Channels:

* simulation   -- run the steps forward, tracking the state of the world.
* causal links -- for every need, trace back to the step (or starting
                  condition) that supplies it, and find steps that threaten
                  that link. A static method, built differently from the
                  simulation; the two must agree.
* timeline     -- dependency layers: what could happen in parallel, and
                  the critical path (a projection; it abstains).
* palace       -- a memory-palace route: one locus per step, with the reason
                  for the ordering attached to the place (a projection).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..scene import Scene
from .base import ChannelView, Domain, Imagination, Status, Verdict

KINDS = {"needs", "makes", "ends"}
LOCI = ["front door", "hallway", "coat rack", "staircase", "landing", "bathroom",
        "bedroom", "window seat", "kitchen table", "stove", "fridge", "back door",
        "garden gate", "garden bench", "shed", "rooftop"]


@dataclass
class Step:
    sid: str
    label: str
    needs: list[tuple[str, str]] = field(default_factory=list)  # (fact, rid)
    makes: list[tuple[str, str]] = field(default_factory=list)
    ends: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Run:
    order: list[str]
    ok: bool
    failures: list[tuple[int, str, list[str], list[str]]]  # (index, step, missing facts, rids)
    unmet_goals: list[str]
    states: list[set[str]]


class ProcessDomain(Domain):
    name = "process"

    # ---------------------------------------------------------------- helpers
    def steps(self, scene: Scene) -> dict[str, Step]:
        out = {sid: Step(sid, scene.label(sid)) for sid in scene.extras.get("steps", [])}
        for r in scene.relations:
            if r.kind in KINDS and r.args[0] in out:
                getattr(out[r.args[0]], r.kind).append((r.args[1], r.rid))
        return out

    def run(self, scene: Scene, order: list[str]) -> Run:
        steps = self.steps(scene)
        state = set(scene.extras.get("initial", []))
        states, failures = [set(state)], []
        for i, sid in enumerate(order):
            st = steps[sid]
            missing = [(f, rid) for f, rid in st.needs if f not in state]
            if missing:
                failures.append((i, sid, [f for f, _ in missing], [rid for _, rid in missing]))
            state -= {f for f, _ in st.ends}
            state |= {f for f, _ in st.makes}
            states.append(set(state))
        unmet = [g for g in scene.extras.get("goal", []) if g not in state]
        return Run(order, not failures and not unmet, failures, unmet, states)

    def repair(self, scene: Scene, budget: int) -> list[str] | None:
        """Find a working order of the same steps, as close to the stated one as possible."""
        steps = self.steps(scene)
        stated = list(scene.extras.get("steps", []))
        goal = set(scene.extras.get("goal", []))
        nodes = 0

        def dfs(state: frozenset, remaining: list[str], path: list[str]):
            nonlocal nodes
            nodes += 1
            if nodes > budget:
                return None
            if not remaining:
                return list(path) if goal <= state else None
            for sid in remaining:  # stated order first => minimal disturbance
                st = steps[sid]
                if all(f in state for f, _ in st.needs):
                    nxt = (state - {f for f, _ in st.ends}) | {f for f, _ in st.makes}
                    found = dfs(frozenset(nxt), [r for r in remaining if r != sid], path + [sid])
                    if found:
                        return found
            return None

        return dfs(frozenset(scene.extras.get("initial", [])), stated, [])

    @staticmethod
    def moves(stated: list[str], fixed: list[str]) -> list[str]:
        """Steps that must move: those outside the longest common subsequence."""
        n, m = len(stated), len(fixed)
        lcs = [[0] * (m + 1) for _ in range(n + 1)]
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                lcs[i][j] = lcs[i + 1][j + 1] + 1 if stated[i] == fixed[j] else \
                    max(lcs[i + 1][j], lcs[i][j + 1])
        keep, i, j = set(), 0, 0
        while i < n and j < m:
            if stated[i] == fixed[j]:
                keep.add(stated[i]); i += 1; j += 1
            elif lcs[i + 1][j] > lcs[i][j + 1]:  # on ties, move the step that must come earlier
                i += 1
            else:
                j += 1
        return [s for s in fixed if s not in keep]

    # ---------------------------------------------------------------- envision
    def envision(self, scene: Scene, budget: int) -> Imagination:
        stated = list(scene.extras.get("steps", []))
        run = self.run(scene, stated)
        im = Imagination(scene=scene, worlds=[run], preferred=run)
        im.data["run"] = run
        im.data["repair"] = None if run.ok else self.repair(scene, budget)
        im.data["inconsistent"] = False
        failing = {sid for _, sid, _, _ in run.failures}
        steps = self.steps(scene)
        for sid in stated:
            needs = steps[sid].needs
            if not needs:
                im.vividness[sid] = 1.0
            else:
                missing = next((len(m) for _, s, m, _ in run.failures if s == sid), 0)
                im.vividness[sid] = 1 - missing / len(needs) if sid in failing else 1.0
        return im

    def glance(self, im: Imagination) -> Status | None:
        # Read at face value, a written procedure looks like it works.
        return Status.WORKS

    # --------------------------------------------------------------- channels
    def refract(self, im: Imagination) -> list[ChannelView]:
        scene = im.scene
        run: Run = im.data["run"]
        views = [self._simulation_view(scene, run), self._causal_view(scene, run.order)]
        best = run.order if run.ok else (im.data["repair"] or run.order)
        views.append(self._timeline_view(scene, best))
        views.append(self._palace_view(scene, best))
        return views

    def _simulation_view(self, scene: Scene, run: Run) -> ChannelView:
        steps = self.steps(scene)
        lines = ["start: " + (", ".join(scene.label(f) for f in sorted(run.states[0])) or "(nothing)")]
        fails = {i: (m, rids) for i, _, m, rids in run.failures}
        for i, sid in enumerate(run.order):
            st = steps[sid]
            mark = "ok " if i not in fails else "FAIL"
            lines.append(f"{i + 1:>2}. [{mark}] {st.label}")
            if i in fails:
                lines.append("        missing: " + ", ".join(scene.label(f) for f in fails[i][0]))
            delta = [f"+{scene.label(f)}" for f, _ in st.makes] + [f"-{scene.label(f)}" for f, _ in st.ends]
            if delta:
                lines.append("        " + " ".join(delta))
        if run.unmet_goals:
            lines.append("goal NOT reached; missing: " + ", ".join(scene.label(g) for g in run.unmet_goals))
        else:
            lines.append("goal reached" if not run.failures else "goal state reached only by pushing through failures")
        conflicts = [rid for _, _, _, rids in run.failures for rid in rids]
        return ChannelView("simulation", "\n".join(lines), Status.WORKS if run.ok else Status.FAILS,
                           "forward simulation", conflicts)

    def links(self, scene: Scene, order: list[str]):
        """Causal links and threats for an ordering (static analysis)."""
        steps = self.steps(scene)
        initial = set(scene.extras.get("initial", []))
        consumers = [(i, sid, f, rid) for i, sid in enumerate(order) for f, rid in steps[sid].needs]
        consumers += [(len(order), "GOAL", g, "goal") for g in scene.extras.get("goal", [])]
        links, broken = [], []
        for i, sid, f, rid in consumers:
            producers = [j for j in range(i) if f in {x for x, _ in steps[order[j]].makes}]
            supplier = producers[-1] if producers else (-1 if f in initial else None)
            if supplier is None:
                broken.append((sid, f, rid, None, "nothing before it makes this true"))
                continue
            threat = next((k for k in range(supplier + 1, i)
                           if f in {x for x, _ in steps[order[k]].ends}), None)
            src = "start" if supplier < 0 else order[supplier]
            if threat is not None:
                broken.append((sid, f, rid, order[threat], f"'{steps[order[threat]].label}' ends it first"))
            else:
                links.append((src, sid, f, rid))
        return links, broken

    def _causal_view(self, scene: Scene, order: list[str]) -> ChannelView:
        links, broken = self.links(scene, order)
        lab = lambda s: "start" if s == "start" else ("GOAL" if s == "GOAL" else scene.label(s))
        lines = [f"  {scene.label(f)}: {lab(src)} --> {lab(dst)}" for src, dst, f, _ in links]
        for dst, f, rid, threat, why in broken:
            lines.append(f"  BROKEN {scene.label(f)} -> {lab(dst)} [{rid}]: {why}")
        conflicts = [rid for _, _, rid, _, _ in broken if rid != "goal"]
        return ChannelView("causal links", "\n".join(lines) or "(no dependencies)",
                           Status.FAILS if broken else Status.WORKS, "causal-link analysis", conflicts)

    def orderings(self, scene: Scene, order: list[str]) -> dict[str, set[str]]:
        """Predecessors each step really needs: its suppliers, plus threat resolution.

        A step that ends fact f must not fall inside any causal link that carries
        f: it goes after the consumer (or before the producer), whichever the
        ordering already chose.
        """
        steps = self.steps(scene)
        links, _ = self.links(scene, order)
        preds: dict[str, set[str]] = {sid: set() for sid in order}
        pos = {sid: i for i, sid in enumerate(order)}
        for src, dst, f, _ in links:
            if src != "start" and dst != "GOAL":
                preds[dst].add(src)
            for k in order:
                if k in (src, dst) or f not in {x for x, _ in steps[k].ends}:
                    continue
                if dst != "GOAL" and pos[k] > pos[dst]:
                    preds[k].add(dst)
                elif src != "start" and pos[k] < pos[src]:
                    preds[src].add(k)
        return preds

    def _timeline_view(self, scene: Scene, order: list[str]) -> ChannelView:
        preds = self.orderings(scene, order)
        layer: dict[str, int] = {}
        for sid in order:  # every constraint points forward in a working order
            layer[sid] = 1 + max((layer[p] for p in preds[sid] if p in layer), default=-1)
        by: dict[int, list[str]] = {}
        for sid, k in layer.items():
            by.setdefault(k, []).append(scene.label(sid))
        lines = [f"  t{k}: " + "  ||  ".join(by[k]) for k in sorted(by)]
        lines.append(f"  critical path: {len(by)} stage(s) for {len(order)} step(s)"
                     + ("; '||' marks steps that could run in parallel" if len(by) < len(order) else ""))
        return ChannelView("timeline", "\n".join(lines), None, "dependency layering", independent=False)

    def _palace_view(self, scene: Scene, order: list[str]) -> ChannelView:
        steps = self.steps(scene)
        links, _ = self.links(scene, order)
        lines = []
        for i, sid in enumerate(order):
            st = steps[sid]
            locus = LOCI[i % len(LOCI)]
            needs = ", ".join(scene.label(f) for f, _ in st.needs) or "nothing"
            leaves = ", ".join(scene.label(f) for f, _ in st.makes) or "nothing new"
            lines.append(f"  {i + 1}. At the {locus}: {st.label}.")
            lines.append(f"       needs {needs}; leaves {leaves}"
                         + (f"; takes away {', '.join(scene.label(f) for f, _ in st.ends)}" if st.ends else ""))
            because = [scene.label(src) for src, dst, _, _ in links if dst == sid and src not in ("start",)]
            if because:
                lines.append(f"       (you can only reach this room after: {', '.join(dict.fromkeys(because))})")
        return ChannelView("palace", "\n".join(lines), None, "method of loci", independent=False)

    def animate(self, im: Imagination) -> list[str]:
        run: Run = im.data["run"]
        events = [f"rehearsed {len(run.order)} step(s) as written: "
                  + ("goal reached" if run.ok else f"{len(run.failures)} step(s) fail")]
        fixed = im.data.get("repair")
        if fixed:
            events.append("rehearsed the repaired order: goal reached")
        return events

    # ----------------------------------------------------------------- settle
    def settle(self, im: Imagination) -> Verdict:
        scene = im.scene
        run: Run = im.data["run"]
        if run.ok:
            return Verdict(Status.WORKS, "The procedure works as written.", witness=run.order)
        fixed = im.data.get("repair")
        problems = []
        for _, sid, missing, rids in run.failures:
            problems.append(f"'{scene.label(sid)}' needs {', '.join(scene.label(f) for f in missing)}"
                            f" [{', '.join(rids)}]")
        if run.unmet_goals:
            problems.append("the goal is not reached (" + ", ".join(scene.label(g) for g in run.unmet_goals) + ")")
        text = "The procedure fails as written: " + "; ".join(problems) + "."
        notes = []
        if fixed:
            moved = self.moves(run.order, fixed)
            for sid in moved:
                k = fixed.index(sid)
                anchor = f"before '{scene.label(fixed[k + 1])}'" if k + 1 < len(fixed) else "to the end"
                notes.append(f"move '{scene.label(sid)}' {anchor}")
            text += " Repair: " + "; ".join(notes) + "."
        else:
            made = {f for st in self.steps(scene).values() for f, _ in st.makes}
            gaps = sorted({f for _, _, m, _ in run.failures for f in m} | set(run.unmet_goals))
            gaps = [g for g in gaps if g not in made and g not in scene.extras.get("initial", [])]
            if gaps:
                notes = [f"a step is missing: nothing ever makes '{scene.label(g)}' true" for g in gaps]
                text += " No reordering helps; " + "; ".join(notes) + "."
            else:
                text += " No reordering of these steps reaches the goal."
        conflicts = [rid for _, _, _, rids in run.failures for rid in rids]
        return Verdict(Status.FAILS, text, counterexample=run.order, conflict_set=conflicts,
                       repair=fixed, notes=notes)

    def render_world(self, im: Imagination, world) -> str:
        order = world.order if isinstance(world, Run) else world
        return " -> ".join(im.scene.label(s) for s in order)

    def signature(self, scene: Scene) -> dict:
        sig = super().signature(scene)
        sig["n_entities"] = len(scene.extras.get("steps", []))
        return sig
