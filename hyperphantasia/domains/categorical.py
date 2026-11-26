"""Categorical imagination: "All A are B", "Some B are C", and what follows.

This is the home ground of Johnson-Laird's mental-model theory. People
imagine a few representative individuals, read a conclusion off that picture,
and (if they are careful) hunt for an alternative picture in which the
conclusion fails. Most syllogism errors come from stopping after the first
picture. Here the first picture is built the way people build it (greedily
merging individuals), and then two independent methods check it:

* individuals -- enumerate every possible population of individual *types*.
* venn        -- shade and mark regions of a Venn diagram (constraint
                 propagation). Different method, same answers when correct.
"""

from __future__ import annotations

from itertools import combinations

from ..scene import Relation, Scene, singular, with_article
from .base import ChannelView, Domain, Imagination, Status, Verdict

KINDS = {"all", "some", "no", "some_not"}
WORD = {"all": "All {} are {}", "some": "Some {} are {}", "no": "No {} are {}",
        "some_not": "Some {} are not {}"}

Type = frozenset  # an individual type: the set of terms it falls under
World = frozenset  # a world: the set of types that have at least one member


def true_in(kind: str, x: str, y: str, world: World) -> bool:
    if kind == "all":
        return not any(x in t and y not in t for t in world)
    if kind == "no":
        return not any(x in t and y in t for t in world)
    if kind == "some":
        return any(x in t and y in t for t in world)
    if kind == "some_not":
        return any(x in t and y not in t for t in world)
    raise ValueError(kind)


class CategoricalDomain(Domain):
    name = "categorical"

    def _terms(self, scene: Scene) -> list[str]:
        return list(scene.entities)

    def _import(self, scene: Scene) -> bool:
        return scene.extras.get("existential_import", True)

    def _satisfies(self, scene: Scene, world: World, relations: list[Relation]) -> bool:
        if self._import(scene) and not all(any(x in t for t in world) for x in self._terms(scene)):
            return False
        return all(true_in(r.kind, *r.args, world) for r in relations)

    # ---------------------------------------------------------------- envision
    def envision(self, scene: Scene, budget: int) -> Imagination:
        terms = self._terms(scene)
        if len(terms) > 4:
            raise ValueError("categorical imagination supports at most 4 terms")
        types = [Type(c) for n in range(1, len(terms) + 1) for c in combinations(terms, n)]
        worlds, complete = [], True
        for n in range(1, len(types) + 1):
            for combo in combinations(types, n):
                w = World(combo)
                if self._satisfies(scene, w, scene.relations):
                    worlds.append(w)
                    if len(worlds) >= budget:
                        complete = False
                        break
            if not complete:
                break
        im = Imagination(scene=scene, worlds=worlds, complete=complete)
        im.data["inconsistent"] = not worlds
        im.preferred = self._preferred(scene) if worlds else None
        im.vividness = self._vividness(scene, worlds)
        return im

    def _preferred(self, scene: Scene) -> World:
        """Build the first picture the way people do: greedily, reusing tokens."""
        rules = [r.args for r in scene.relations if r.kind == "all"]
        noes = [set(r.args) for r in scene.relations if r.kind == "no"]

        def close(t: set[str]) -> set[str]:
            t = set(t)
            grew = True
            while grew:
                grew = False
                for x, y in rules:
                    if x in t and y not in t:
                        t.add(y)
                        grew = True
            return t

        def clashes(t: set[str]) -> bool:
            return any(n <= t for n in noes)

        people: list[set[str]] = []
        for r in scene.relations:
            x, y = r.args
            if r.kind == "all" and not any(x in p for p in people):
                people.append(close({x}))
            elif r.kind == "some":
                for p in people:  # the tempting shortcut: merge into someone we already have
                    if (x in p or y in p) and not clashes(close(p | {x, y})):
                        p |= close({x, y})
                        break
                else:
                    people.append(close({x, y}))
            elif r.kind == "some_not":
                if not any(x in p and y not in p for p in people):
                    people.append(close({x}))
            elif r.kind == "no":
                for z in (x, y):
                    if not any(z in p for p in people):
                        people.append(close({z}))
        if self._import(scene):
            for z in self._terms(scene):
                if not any(z in p for p in people):
                    people.append(close({z}))
        world = World(Type(close(p)) for p in people)
        if self._satisfies(scene, world, scene.relations):
            return world
        # The greedy picture broke a premise; fall back to the simplest valid one.
        return min((w for w in self._all_valid(scene)), key=len)

    def _all_valid(self, scene: Scene):
        terms = self._terms(scene)
        types = [Type(c) for n in range(1, len(terms) + 1) for c in combinations(terms, n)]
        for n in range(1, len(types) + 1):
            for combo in combinations(types, n):
                w = World(combo)
                if self._satisfies(scene, w, scene.relations):
                    yield w

    def _vividness(self, scene: Scene, worlds: list[World]) -> dict[str, float]:
        terms = self._terms(scene)
        out = {}
        for x in terms:
            others = [y for y in terms if y != x]
            if not others or not worlds:
                out[x] = 1.0
                continue
            fixed = 0
            for y in others:
                profiles = {tuple(true_in(k, x, y, w) for k in sorted(KINDS)) for w in worlds}
                fixed += len(profiles) == 1
            out[x] = fixed / len(others)
        return out

    # ------------------------------------------------------------------ query
    def glance(self, im: Imagination) -> Status | None:
        q = im.scene.query
        if not q or im.preferred is None:
            return None
        return Status.NECESSARY if true_in(q.kind, *q.args, im.preferred) else Status.IMPOSSIBLE

    def _status(self, kind: str, x: str, y: str, worlds: list[World]):
        trues = [w for w in worlds if true_in(kind, x, y, w)]
        falses = sorted((w for w in worlds if not true_in(kind, x, y, w)), key=len)
        if not falses:
            return Status.NECESSARY, (min(trues, key=len) if trues else None), None
        if not trues:
            return Status.IMPOSSIBLE, None, falses[0]
        return Status.POSSIBLE, min(trues, key=len), falses[0]

    # --------------------------------------------------------------- channels
    def refract(self, im: Imagination) -> list[ChannelView]:
        scene = im.scene
        q = scene.query
        views = [ChannelView("picture", self.render_world(im, im.preferred) if im.preferred
                             else "(no consistent picture can be formed)",
                             self.glance(im), "one greedily-built population", independent=False)]
        if im.data["inconsistent"]:
            ind_verdict = Status.INCONSISTENT
            body = "no population of individuals satisfies every premise"
        else:
            ind_verdict = self._status(q.kind, *q.args, im.worlds)[0] if q else Status.OPEN
            body = f"{len(im.worlds)} distinct population(s) satisfy the premises" + \
                   ("" if im.complete else " (budget reached; partial)")
        views.append(ChannelView("individuals", body, ind_verdict, "exhaustive population enumeration",
                                 needs_complete=True))
        views.append(self._venn(im))
        return views

    def _venn(self, im: Imagination) -> ChannelView:
        scene = im.scene
        terms = self._terms(scene)
        regions = [Type(c) for n in range(1, len(terms) + 1) for c in combinations(terms, n)]
        shaded: dict[Type, list[str]] = {}
        for r in scene.relations:
            x, y = r.args
            for reg in regions:
                if (r.kind == "all" and x in reg and y not in reg) or \
                   (r.kind == "no" and x in reg and y in reg):
                    shaded.setdefault(reg, []).append(r.rid)
        marks: list[tuple[str, set[Type]]] = []
        for r in scene.relations:
            x, y = r.args
            if r.kind == "some":
                marks.append((r.rid, {g for g in regions if x in g and y in g and g not in shaded}))
            elif r.kind == "some_not":
                marks.append((r.rid, {g for g in regions if x in g and y not in g and g not in shaded}))
        if self._import(scene):
            for z in terms:
                marks.append((f"exists:{scene.label(z)}", {g for g in regions if z in g and g not in shaded}))
        lines = []
        for reg in sorted(regions, key=lambda g: (-len(g), sorted(g))):
            name = " & ".join(scene.label(t) if t in reg else f"not {scene.label(t)}" for t in terms)
            if reg in shaded:
                state = f"empty ({', '.join(shaded[reg])})"
            else:
                ids = [m for m, regs in marks if reg in regs]
                state = ("x? " + ", ".join(ids)) if ids else "?"
            lines.append(f"  {name:<{14 * len(terms)}} {state}")
        lines.append("  (x? = someone is in one of the regions carrying that mark)")
        if any(not regs for _, regs in marks):
            empty = [m for m, regs in marks if not regs]
            return ChannelView("venn", "\n".join(lines), Status.INCONSISTENT,
                               "region shading", conflicts=empty)
        verdict = None
        q = scene.query
        if q:
            x, y = q.args
            if q.kind in ("all", "no"):
                target = {g for g in regions if x in g and ((y not in g) if q.kind == "all" else (y in g))}
                if target <= shaded.keys():
                    verdict = Status.NECESSARY
                elif any(regs <= target for _, regs in marks):
                    verdict = Status.IMPOSSIBLE
                else:
                    verdict = Status.POSSIBLE
            else:
                target = {g for g in regions if x in g and ((y in g) if q.kind == "some" else (y not in g))}
                if any(regs <= target for _, regs in marks):
                    verdict = Status.NECESSARY
                elif target <= shaded.keys():
                    verdict = Status.IMPOSSIBLE
                else:
                    verdict = Status.POSSIBLE
        return ChannelView("venn", "\n".join(lines), verdict, "region shading")

    # ----------------------------------------------------------------- settle
    def settle(self, im: Imagination) -> Verdict:
        scene = im.scene
        if im.data["inconsistent"]:
            core = list(scene.relations)
            for r in list(core):
                trial = [c for c in core if c is not r]
                if not any(self._satisfies(scene, w, trial) for w in self._universe(scene)):
                    core = trial
            ids = [r.rid for r in core]
            names = "; ".join(f"[{r.rid}] {r.source}" for r in core)
            return Verdict(Status.INCONSISTENT, f"The premises cannot all be true together: {names}",
                           conflict_set=ids)
        q = scene.query
        if not q:
            follows = self.what_follows(im)
            text = ("What follows: " + "; ".join(follows) + ".") if follows else \
                   "Nothing new follows about the terms that the premises do not already say."
            return Verdict(Status.OPEN, text, im.preferred, notes=follows)
        status, witness, counter = self._status(q.kind, *q.args, im.worlds)
        claim = WORD[q.kind].format(scene.label(q.args[0]), scene.label(q.args[1])).lower()
        text = {
            Status.NECESSARY: f"Yes, it follows: in every population the premises allow, {claim}.",
            Status.IMPOSSIBLE: f"No: the premises rule out that {claim}.",
            Status.POSSIBLE: f"It does not follow. It could be that {claim}, but the premises "
                             f"also allow a population where it is false.",
        }[status]
        return Verdict(status, text, witness, counter)

    def _universe(self, scene: Scene):
        terms = self._terms(scene)
        types = [Type(c) for n in range(1, len(terms) + 1) for c in combinations(terms, n)]
        for n in range(1, len(types) + 1):
            for combo in combinations(types, n):
                yield World(combo)

    def what_follows(self, im: Imagination) -> list[str]:
        """Strongest new conclusions linking terms no single premise links."""
        scene = im.scene
        linked = {frozenset(r.args) for r in scene.relations}
        necessary = set()
        for x in scene.entities:
            for y in scene.entities:
                if x != y and frozenset((x, y)) not in linked:
                    for kind in KINDS:
                        if self._status(kind, x, y, im.worlds)[0] is Status.NECESSARY:
                            necessary.add((kind, x, y))
        out, seen = [], set()
        for kind, x, y in sorted(necessary, key=lambda k: (["all", "no", "some", "some_not"].index(k[0]),
                                                           list(scene.entities).index(k[1]))):
            if kind == "some" and ({("all", x, y), ("all", y, x)} & necessary):
                continue  # implied by the stronger universal
            if kind == "some_not" and ("no", x, y) in necessary:
                continue
            key = (kind, frozenset((x, y))) if kind in ("some", "no") else (kind, x, y)
            if key not in seen:
                seen.add(key)
                out.append(WORD[kind].format(scene.label(x), scene.label(y)))
        return out

    # -------------------------------------------------------------- rendering
    def verbal(self, im: Imagination, world: World | None) -> str:
        if not world:
            return "(empty)"
        scene = im.scene
        order = list(scene.entities)
        kinds = []
        for ty in sorted(world, key=lambda t: (-len(t), sorted(t, key=order.index))):
            nouns = [with_article(singular(scene.label(t))) for t in sorted(ty, key=order.index)]
            joined = nouns[0] if len(nouns) == 1 else ", ".join(nouns[:-1]) + " and " + nouns[-1]
            kinds.append("someone who is " + joined)
        return "There is " + "; there is ".join(kinds) + "; nobody else."

    def render_world(self, im: Imagination, world: World | None) -> str:
        if not world:
            return "(empty)"
        scene = im.scene
        terms = self._terms(scene)
        width = max(len(scene.label(t)) for t in terms) + 2
        rows = []
        for t in sorted(world, key=lambda t: [terms.index(x) for x in sorted(t, key=terms.index)]):
            rows.append("  " + "".join((scene.label(x) if x in t else "").ljust(width) for x in terms).rstrip())
        return "\n".join(rows) + "\n  (each row is one kind of individual that exists)"
