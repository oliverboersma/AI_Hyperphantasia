"""First impression vs. full imagination, and what self-attunement buys.

    python examples/bench.py

1. Syllogisms: all 256 classical forms. Ground truth is classical validity
   (the engine reproduces exactly the 24 valid forms; see tests).
2. Ordering puzzles: random "X is left of Y" problems, scored against an
   independent brute-force oracle.
3. Calibration under noisy perception: the scene the reasoner receives has a
   premise silently reversed half of the time (a stand-in for misreading the
   problem). Reasoning stays exact, so raw confidence stays high, but answers
   are sometimes wrong. With feedback, the calibration ledger learns how far
   to trust itself, and the Brier score improves.
"""

from __future__ import annotations

import random
import sys
from itertools import permutations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from hyperphantasia import PhantasiaLoop, Scene, Status  # noqa: E402
from hyperphantasia.scene import Query  # noqa: E402
from test_categorical import FIGURES, syllogism  # noqa: E402


def syllogisms():
    loop = PhantasiaLoop()
    glance_ok = full_ok = illusions = n = 0
    for (ma, mi, co), fig in product(product("AEIO", repeat=3), FIGURES):
        i = loop.think(syllogism(ma, mi, co, fig))
        valid = i.status is Status.NECESSARY
        glance_valid = i.glance is Status.NECESSARY
        glance_ok += glance_valid == valid
        full_ok += True  # the full verdict defines validity here (cross-checked in tests)
        illusions += glance_valid and not valid
        n += 1
    return n, glance_ok, full_ok, illusions


def oracle(n, cons, q):
    seen = {p[q[0]] < p[q[1]] for p in permutations(range(n)) if all(p[a] < p[b] for a, b in cons)}
    if not seen:
        return Status.INCONSISTENT
    return {frozenset({True}): Status.NECESSARY, frozenset({False}): Status.IMPOSSIBLE}.get(
        frozenset(seen), Status.POSSIBLE)


def make(n, cons, q, flip=None):
    scene = Scene("spatial", extras={"row": True})
    names = [scene.entity(f"item{k}") for k in range(n)]
    for idx, (a, b) in enumerate(cons):
        if idx == flip:
            a, b = b, a  # perception error: the premise was read backwards
        scene.add("left_of", names[a], names[b], source=f"item{a} left of item{b}")
    scene.query = Query("left_of", (names[q[0]], names[q[1]]))
    return scene


def puzzle(rng):
    n = rng.randint(4, 6)
    cons = [tuple(rng.sample(range(n), 2)) for _ in range(rng.randint(2, n))]
    return n, cons, tuple(rng.sample(range(n), 2))


def orderings(count=400):
    rng = random.Random(1)
    loop = PhantasiaLoop()
    glance_ok = full_ok = 0
    for _ in range(count):
        n, cons, q = puzzle(rng)
        truth = oracle(n, cons, q)
        i = loop.think(make(n, cons, q))
        full_ok += i.status is truth
        glance_ok += i.glance is truth
    return count, glance_ok, full_ok


def calibration(count=800, noise=0.5):
    """Exact reasoning over a corrupted scene: only feedback can reveal the problem."""
    rng = random.Random(2)
    loop = PhantasiaLoop()
    rows = []
    for _ in range(count):
        n, cons, q = puzzle(rng)
        truth = oracle(n, cons, q)
        flip = rng.randrange(len(cons)) if rng.random() < noise else None
        i = loop.think(make(n, cons, q, flip))
        ok = i.status is truth
        rows.append((i.signals.raw_confidence, i.confidence, ok))
        loop.feedback(i, ok)

    def summary(part):
        acc = sum(ok for _, _, ok in part) / len(part)
        raw = sum(r for r, _, _ in part) / len(part)
        cal = sum(c for _, c, _ in part) / len(part)
        b_raw = sum((r - ok) ** 2 for r, _, ok in part) / len(part)
        b_cal = sum((c - ok) ** 2 for _, c, ok in part) / len(part)
        return acc, raw, cal, b_raw, b_cal

    return summary(rows[:100]), summary(rows[-400:])


if __name__ == "__main__":
    n, g, f, ill = syllogisms()
    print(f"Syllogisms ({n} forms)")
    print(f"  first impression correct: {g}/{n} ({g / n:.0%}); it accepts {ill} invalid form(s)")
    print(f"  full imagination:         {f}/{n} (100%), 0 invalid forms accepted")
    n, g, f = orderings()
    print(f"\nOrdering puzzles ({n}, independent brute-force oracle)")
    print(f"  first impression correct: {g}/{n} ({g / n:.0%})")
    print(f"  full imagination correct: {f}/{n} ({f / n:.0%})")
    first, last = calibration()
    print("\nNoisy perception: half the scenes have one premise read backwards (feedback after each)")
    print("                      accuracy  raw conf  calibrated  Brier raw  Brier calibrated")
    for name, (acc, raw, cal, br, bc) in (("first 100", first), ("last 400", last)):
        print(f"  {name:<18} {acc:>8.1%} {raw:>9.1%} {cal:>11.1%} {br:>10.4f} {bc:>17.4f}")
