"""Spatial imagination: objects on a line or a 2-D grid.

Three channels look at the same scene:

* picture      -- the single preferred image (the "first thing that comes to
                  mind"). Fast, vivid, and sometimes wrong.
* possibilities-- every arrangement consistent with the premises (a
                  Johnson-Laird style model space). Semantic and exhaustive.
* relations    -- a transitive-closure graph over the premises. Symbolic,
                  proof-producing, and blind to disjunctive premises.

The last two are built by different methods, so when both commit to an answer
and disagree, something is wrong with the reasoner itself (dissonance).
When the picture disagrees with the possibility space, the reasoner has caught
its own first impression being misleading (surprise), which is a lesson worth
keeping.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from itertools import combinations
from typing import Iterator

from ..scene import Relation, Scene
from .base import ChannelView, Domain, Imagination, Status, Verdict

AXES = ("x", "y")  # x grows to the right, y grows upward
DISJUNCTIVE = {"adj", "imm", "btw"}

# relation kind -> list of atoms; an atom is (op, axis, *entity_positions)
# where entity positions index into the relation's args.
EXPANSION: dict[str, list[tuple]] = {
    "left_of": [("lt", "x", 0, 1)],
    "right_of": [("lt", "x", 1, 0)],
    "above": [("lt", "y", 1, 0)],
    "below": [("lt", "y", 0, 1)],
    "directly_above": [("lt", "y", 1, 0), ("eq", "x", 0, 1)],
    "directly_below": [("lt", "y", 0, 1), ("eq", "x", 0, 1)],
    "directly_left_of": [("lt", "x", 0, 1), ("eq", "y", 0, 1)],
    "directly_right_of": [("lt", "x", 1, 0), ("eq", "y", 0, 1)],
    "same_column": [("eq", "x", 0, 1)],
    "same_row": [("eq", "y", 0, 1)],
    "immediately_left_of": [("imm", "x", 0, 1)],
    "immediately_right_of": [("imm", "x", 1, 0)],
    "next_to": [("adj", "x", 0, 1)],
    "between": [("btw", "x", 0, 1, 2)],
}
SLOT_KINDS = {"immediately_left_of", "immediately_right_of", "next_to", "between"}
KINDS = set(EXPANSION) | {"relative"}

PHRASE = {
    "left_of": "is left of", "right_of": "is right of", "above": "is above",
    "below": "is below", "directly_above": "is directly above",
    "directly_below": "is directly below", "directly_left_of": "is directly left of",
    "directly_right_of": "is directly right of", "same_column": "is in the same column as",
    "same_row": "is in the same row as", "immediately_left_of": "is immediately left of",
    "immediately_right_of": "is immediately right of", "next_to": "is next to",
    "between": "is between",
}


@dataclass(frozen=True)
class Atom:
    op: str
    axis: str
    ents: tuple[str, ...]
    rid: str = ""


def atoms_of(kind: str, args: tuple[str, ...], rid: str = "") -> list[Atom]:
    return [Atom(op, axis, tuple(args[i] for i in idx), rid)
            for op, axis, *idx in EXPANSION[kind]]


def holds(atom: Atom, pos: dict[str, int]) -> bool:
    v = [pos[e] for e in atom.ents]
    if atom.op == "lt":
        return v[0] < v[1]
    if atom.op == "eq":
        return v[0] == v[1]
    if atom.op == "imm":
        return v[1] - v[0] == 1
    if atom.op == "adj":
        return abs(v[0] - v[1]) == 1
    if atom.op == "btw":
        return v[1] < v[0] < v[2] or v[2] < v[0] < v[1]
    raise ValueError(atom.op)


class AxisSpace:
    """All placements of some entities along one axis.

    In ``slots`` mode every entity occupies its own position (a row of seats).
    Otherwise placements are weak orders: entities may share a coordinate.
    """

    def __init__(self, axis: str, ents: list[str], atoms: list[Atom], slots: bool):
        self.axis, self.ents, self.atoms, self.slots = axis, ents, atoms, slots
        self.preds: dict[str, set[str]] = {e: set() for e in ents}
        self.eqs: list[Atom] = []
        for a in atoms:
            if a.op == "lt":
                self.preds[a.ents[1]].add(a.ents[0])
            elif a.op == "eq":
                self.eqs.append(a)

    def _leaf_ok(self, pos: dict[str, int]) -> bool:
        return all(holds(a, pos) for a in self.atoms)

    def worlds(self, limit: int) -> Iterator[dict[str, int]]:
        """Yield placements in 'preferred' order: crisp, mention-ordered first."""
        count = 0
        order = {e: i for i, e in enumerate(self.ents)}

        def weak(remaining: list[str], level: int, pos: dict[str, int]):
            nonlocal count
            if count >= limit:
                return
            if not remaining:
                if self._leaf_ok(pos):
                    count += 1
                    yield dict(pos)
                return
            if any((a.ents[0] in pos) != (a.ents[1] in pos) for a in self.eqs):
                return  # one side of an equality was placed without the other
            avail = [e for e in remaining if self.preds[e] <= pos.keys()]
            for size in range(1, len(avail) + 1):
                for group in combinations(avail, size):
                    gs = set(group)
                    if any((a.ents[0] in gs) != (a.ents[1] in gs) for a in self.eqs
                           if a.ents[0] in remaining and a.ents[1] in remaining):
                        continue
                    for e in group:
                        pos[e] = level
                    rest = [e for e in remaining if e not in gs]
                    yield from weak(rest, level + 1, pos)
                    for e in group:
                        del pos[e]
                    if count >= limit:
                        return

        def perm(remaining: list[str], slot: int, pos: dict[str, int]):
            nonlocal count
            if count >= limit:
                return
            if not remaining:
                if self._leaf_ok(pos):
                    count += 1
                    yield dict(pos)
                return
            for e in sorted(remaining, key=order.get):
                if not self.preds[e] <= pos.keys():
                    continue
                pos[e] = slot
                yield from perm([r for r in remaining if r != e], slot + 1, pos)
                del pos[e]
                if count >= limit:
                    return

        gen = perm if self.slots else weak
        yield from gen(list(self.ents), 0, {})


def _axis_entities(scene: Scene, atoms: list[Atom], axis: str, slots: bool) -> list[str]:
    used = {e for a in atoms if a.axis == axis for e in a.ents}
    if scene.query and scene.query.kind in EXPANSION:
        for a in atoms_of(scene.query.kind, scene.query.args):
            if a.axis == axis:
                used.update(a.ents)
    if scene.query and scene.query.kind == "relative" and used & set(scene.query.args):
        used.update(scene.query.args)
    if axis == "x" and slots:
        used = set(scene.entities)
    return [e for e in scene.entities if e in used]  # keep mention order


class SpatialDomain(Domain):
    name = "spatial"

    # ---------------------------------------------------------------- envision
    def _atoms(self, relations: list[Relation]) -> list[Atom]:
        out: list[Atom] = []
        for r in relations:
            out.extend(atoms_of(r.kind, r.args, r.rid))
        return out

    def _spaces(self, scene: Scene, relations: list[Relation]) -> dict[str, AxisSpace]:
        slots = bool(scene.extras.get("row")) or any(r.kind in SLOT_KINDS for r in relations)
        atoms = self._atoms(relations)
        spaces = {}
        for axis in AXES:
            ents = _axis_entities(scene, atoms, axis, slots and axis == "x")
            if ents:
                spaces[axis] = AxisSpace(axis, ents, [a for a in atoms if a.axis == axis],
                                         slots and axis == "x")
        return spaces

    def _satisfiable(self, scene: Scene, relations: list[Relation]) -> bool:
        return all(next(sp.worlds(1), None) is not None
                   for sp in self._spaces(scene, relations).values())

    def envision(self, scene: Scene, budget: int) -> Imagination:
        spaces = self._spaces(scene, scene.relations)
        worlds: dict[str, list[dict[str, int]]] = {}
        complete = True
        for axis, sp in spaces.items():
            worlds[axis] = list(sp.worlds(budget))
            if len(worlds[axis]) >= budget:
                complete = False
        im = Imagination(scene=scene, worlds=worlds, complete=complete)
        im.data["spaces"] = spaces
        im.data["inconsistent"] = any(not w for w in worlds.values())
        if not im.data["inconsistent"]:
            im.preferred = {axis: ws[0] for axis, ws in worlds.items()}
        im.vividness = self._vividness(worlds)
        return im

    @staticmethod
    def _vividness(worlds: dict[str, list[dict[str, int]]]) -> dict[str, float]:
        """How fixed is each entity relative to the others (1 = crisp, 0 = blur)."""
        per: dict[str, list[float]] = {}
        for ws in worlds.values():
            if not ws:
                continue
            ents = list(ws[0])
            for e in ents:
                others = [o for o in ents if o != e]
                if not others:
                    continue
                fixed = 0
                for o in others:
                    signs = {(w[e] > w[o]) - (w[e] < w[o]) for w in ws}
                    fixed += len(signs) == 1
                per.setdefault(e, []).append(fixed / len(others))
        return {e: sum(v) / len(v) for e, v in per.items()}

    # ------------------------------------------------------------------ query
    def _query_atoms(self, scene: Scene) -> list[Atom]:
        q = scene.query
        if not q or q.kind not in EXPANSION:
            return []
        return atoms_of(q.kind, q.args)

    def _eval_world(self, atoms: list[Atom], world: dict[str, dict[str, int]]) -> bool:
        return all(holds(a, world[a.axis]) for a in atoms)

    def glance(self, im: Imagination) -> Status | None:
        if im.data["inconsistent"]:
            return None  # no picture forms at all
        atoms = self._query_atoms(im.scene)
        if not atoms:
            return None
        return Status.NECESSARY if self._eval_world(atoms, im.preferred) else Status.IMPOSSIBLE

    def _space_status(self, im: Imagination) -> tuple[Status, dict | None, dict | None]:
        """Evaluate the query over the whole possibility space."""
        atoms = self._query_atoms(im.scene)
        by_axis: dict[str, list[Atom]] = {}
        for a in atoms:
            by_axis.setdefault(a.axis, []).append(a)
        witness = dict(im.preferred)
        counter = None
        all_true = True
        for axis, axatoms in by_axis.items():
            ws = im.worlds[axis]
            true_w = next((w for w in ws if all(holds(a, w) for a in axatoms)), None)
            false_w = next((w for w in ws if not all(holds(a, w) for a in axatoms)), None)
            if true_w is None:
                counter = dict(im.preferred, **{axis: ws[0]})
                return Status.IMPOSSIBLE, None, counter
            witness[axis] = true_w
            if false_w is not None:
                all_true = False
                counter = dict(im.preferred, **{axis: false_w})
        if all_true:
            return Status.NECESSARY, witness, None
        return Status.POSSIBLE, witness, counter

    # --------------------------------------------------------------- channels
    def refract(self, im: Imagination) -> list[ChannelView]:
        scene = im.scene
        views = [
            ChannelView("picture", self.render_world(im, im.preferred) if im.preferred
                        else "(no consistent picture can be formed)",
                        self.glance(im), "one preferred arrangement", independent=False),
        ]
        # possibility space
        if im.data["inconsistent"]:
            poss_verdict = Status.INCONSISTENT
            lines = ["no arrangement satisfies every premise"]
        else:
            if not scene.query:
                poss_verdict = Status.OPEN
            elif scene.query.kind == "relative":
                poss_verdict = self._relative(im).status
            else:
                poss_verdict = self._space_status(im)[0]
            lines = []
            for axis, ws in im.worlds.items():
                tag = "" if im.complete else " (budget reached; partial)"
                lines.append(f"{axis}-axis: {len(ws)} arrangement(s){tag}")
                for w in ws[:4]:
                    lines.append("   " + self._order_string(scene, w, axis))
                if len(ws) > 4:
                    lines.append(f"   ... and {len(ws) - 4} more")
        views.append(ChannelView("possibilities", "\n".join(lines), poss_verdict,
                                 "exhaustive model enumeration", needs_complete=True))
        views.append(self._relational_view(im))
        return views

    def _order_string(self, scene: Scene, world: dict[str, int], axis: str) -> str:
        levels: dict[int, list[str]] = {}
        for e, v in world.items():
            levels.setdefault(v, []).append(scene.label(e))
        arrow = "left -> right" if axis == "x" else "bottom -> top"
        parts = [("=".join(levels[k]) if len(levels[k]) == 1 else "(" + " = ".join(levels[k]) + ")")
                 for k in sorted(levels)]
        return "  <  ".join(parts) + f"    [{arrow}]"

    def _closure(self, ents: list[str], atoms: list[Atom]):
        rel: dict[tuple[str, str], str] = {(e, e): "=" for e in ents}
        for a in atoms:
            if a.op == "lt":
                rel[a.ents] = "<"
            elif a.op == "eq":
                rel.setdefault(a.ents, "=")
                rel.setdefault(a.ents[::-1], "=")
        changed = True
        while changed:
            changed = False
            for k in ents:
                for i in ents:
                    ik = rel.get((i, k))
                    if not ik:
                        continue
                    for j in ents:
                        kj = rel.get((k, j))
                        if not kj:
                            continue
                        comp = "<" if "<" in (ik, kj) else "="
                        if rel.get((i, j)) != comp and rel.get((i, j)) != "<":
                            rel[(i, j)] = comp
                            changed = True
        contradiction = [e for e in ents if rel.get((e, e)) == "<"]
        return rel, contradiction

    def _chain(self, scene: Scene, atoms: list[Atom], a: str, b: str) -> str:
        """Shortest premise path a -> b, as a readable proof."""
        edges: dict[str, list[tuple[str, str, str]]] = {}
        for at in atoms:
            if at.op == "lt":
                edges.setdefault(at.ents[0], []).append((at.ents[1], "<", at.rid))
            elif at.op == "eq":
                edges.setdefault(at.ents[0], []).append((at.ents[1], "=", at.rid))
                edges.setdefault(at.ents[1], []).append((at.ents[0], "=", at.rid))
        prev: dict[str, tuple[str, str, str]] = {}
        seen, dq = {a}, deque([a])
        while dq:
            n = dq.popleft()
            for m, op, rid in edges.get(n, []):
                if m not in seen:
                    seen.add(m)
                    prev[m] = (n, op, rid)
                    dq.append(m)
        if b not in prev:
            return ""
        steps, cur = [], b
        while cur != a:
            p, op, rid = prev[cur]
            steps.append((p, op, cur, rid))
            cur = p
        steps.reverse()
        text = scene.label(a)
        for _, op, m, rid in steps:
            text += f" {op} {scene.label(m)}"
        return text + "   (via " + ", ".join(s[3] for s in steps) + ")"

    def _relational_view(self, im: Imagination) -> ChannelView:
        scene = im.scene
        atoms = self._atoms(scene.relations)
        lines, verdicts, conflicts = [], [], []
        skipped = [a.rid for a in atoms if a.op in DISJUNCTIVE]
        for axis, sp in im.data["spaces"].items():
            ax_atoms = [a for a in atoms if a.axis == axis and a.op in ("lt", "eq")]
            rel, contra = self._closure(sp.ents, ax_atoms)
            if contra:
                conflicts.extend(sorted({a.rid for a in ax_atoms}))
                lines.append(f"{axis}-axis: cycle through {', '.join(scene.label(e) for e in contra)}")
                verdicts.append(Status.INCONSISTENT)
                continue
            im.data.setdefault("closure", {})[axis] = rel
            derived = [(i, j) for (i, j), op in rel.items() if op == "<"]
            lines.append(f"{axis}-axis: {len(derived)} strict relation(s) entailed")
        if skipped:
            lines.append(f"(cannot represent disjunctive premises {', '.join(sorted(set(skipped)))}; abstains on them)")
        if Status.INCONSISTENT in verdicts:
            return ChannelView("relations", "\n".join(lines), Status.INCONSISTENT,
                               "transitive closure", conflicts)
        verdict = None
        q_atoms = self._query_atoms(scene)
        if scene.query and scene.query.kind == "relative":
            a, b = scene.query.args
            fixed = []
            for axis, rel in im.data.get("closure", {}).items():
                if a in im.data["spaces"][axis].ents and b in im.data["spaces"][axis].ents:
                    fixed.append(rel.get((a, b)) is not None or rel.get((b, a)) is not None)
            if fixed and all(fixed):
                verdict = Status.NECESSARY  # every relevant axis is pinned down by the premises
        if q_atoms:
            states = []
            for qa in q_atoms:
                rel = im.data.get("closure", {}).get(qa.axis, {})
                if qa.op == "lt":
                    a, b = qa.ents
                    if rel.get((a, b)) == "<":
                        states.append(True)
                        im.data.setdefault("proofs", []).append(
                            self._chain(scene, [x for x in atoms if x.axis == qa.axis], a, b))
                    elif rel.get((b, a)) in ("<", "=") or rel.get((a, b)) == "=":
                        states.append(False)
                        im.data.setdefault("proofs", []).append(
                            self._chain(scene, [x for x in atoms if x.axis == qa.axis], b, a))
                    else:
                        states.append(None)
                elif qa.op == "eq":
                    a, b = qa.ents
                    if rel.get((a, b)) == "=":
                        states.append(True)
                    elif "<" in (rel.get((a, b)), rel.get((b, a))):
                        states.append(False)
                    else:
                        states.append(None)
                else:
                    states.append(None)
            if all(s is True for s in states):
                verdict = Status.NECESSARY
            elif any(s is False for s in states):
                verdict = Status.IMPOSSIBLE
        for p in im.data.get("proofs", []):
            if p:
                lines.append("proof: " + p)
        return ChannelView("relations", "\n".join(lines), verdict, "transitive closure")

    # ----------------------------------------------------------------- settle
    def settle(self, im: Imagination) -> Verdict:
        scene = im.scene
        if im.data["inconsistent"]:
            mus = self._minimal_conflict(scene)
            names = "; ".join(f"[{r.rid}] {r.source}" for r in scene.relations if r.rid in mus)
            return Verdict(Status.INCONSISTENT, f"The premises cannot all be true together: {names}",
                           conflict_set=mus)
        q = scene.query
        if not scene.entities:
            return Verdict(Status.OPEN, "Nothing to imagine: no objects or relations were recognised.")
        if not q:
            return Verdict(Status.OPEN, "Scene imagined; nothing was asked.", witness=im.preferred)
        if q.kind == "relative":
            return self._relative(im)
        status, witness, counter = self._space_status(im)
        a = scene.label(q.args[0])
        rest = " and ".join(scene.label(x) for x in q.args[1:])
        claim = f"{a} {PHRASE[q.kind]} {rest}"
        text = {
            Status.NECESSARY: f"Yes. In every arrangement the premises allow, {claim}.",
            Status.IMPOSSIBLE: f"No. In no arrangement the premises allow is it true that {claim}.",
            Status.POSSIBLE: f"Not determined. It is possible that {claim}, but not necessary.",
        }[status]
        return Verdict(status, text, witness, counter)

    def _relative(self, im: Imagination) -> Verdict:
        scene = im.scene
        a, b = scene.query.args
        parts, determined = [], True
        words = {"x": ("left of", "right of", "in the same column as"),
                 "y": ("below", "above", "in the same row as")}
        for axis, ws in im.worlds.items():
            if a not in ws[0] or b not in ws[0]:
                continue
            signs = {(w[a] > w[b]) - (w[a] < w[b]) for w in ws}
            if len(signs) == 1:
                s = signs.pop()
                parts.append(f"{words[axis][0] if s < 0 else words[axis][1] if s > 0 else words[axis][2]} "
                             f"{scene.label(b)}")
            else:
                determined = False
                opts = sorted(words[axis][0 if s < 0 else 1 if s > 0 else 2] for s in signs)
                parts.append(f"undetermined {'horizontally' if axis == 'x' else 'vertically'} "
                             f"({' / '.join(opts)})")
        text = f"{scene.label(a)} is " + "; ".join(parts) + "." if parts else "No spatial relation known."
        return Verdict(Status.NECESSARY if determined else Status.POSSIBLE, text, im.preferred)

    def _minimal_conflict(self, scene: Scene) -> list[str]:
        """Deletion-based minimal unsatisfiable subset of the premises."""
        core = list(scene.relations)
        for r in list(core):
            trial = [x for x in core if x is not r]
            if not self._satisfiable(scene, trial):
                core = trial
        return [r.rid for r in core]

    # -------------------------------------------------------------- rendering
    def verbal(self, im: Imagination, world: dict | None) -> str:
        if not world:
            return "(empty)"
        return "\n".join(self._order_string(im.scene, w, axis) for axis, w in world.items())

    def render_world(self, im: Imagination, world: dict | None) -> str:
        if not world:
            return "(empty)"
        scene = im.scene
        xs, ys = world.get("x", {}), world.get("y", {})
        ents = list(scene.entities)
        def col(e):
            return xs.get(e, max(xs.values(), default=-1) + 1)
        def row(e):  # things with no vertical information rest on the ground row
            return ys.get(e, min(ys.values())) if ys else 0
        cells: dict[tuple[int, int], list[str]] = {}
        for e in ents:
            if e not in xs and e not in ys:
                continue
            blurry = (im.vividness.get(e, 1.0) < 0.999 or (xs and e not in xs)
                      or (ys and e not in ys))
            mark = "~" if blurry else ""
            cells.setdefault((row(e), col(e)), []).append(scene.label(e) + mark)
        if not cells:
            return "(empty)"
        rows = sorted({r for r, _ in cells}, reverse=True)
        cols = sorted({c for _, c in cells})
        width = max(len("/".join(v)) for v in cells.values()) + 2
        sep = "+" + "+".join("-" * width for _ in cols) + "+"
        out = [sep]
        for r in rows:
            out.append("|" + "|".join("/".join(cells.get((r, c), [])).center(width) for c in cols) + "|")
            out.append(sep)
        if any(c.endswith("~") for v in cells.values() for c in v):
            out.append("~ = blurry: position not fixed by the premises")
        return "\n".join(out)
