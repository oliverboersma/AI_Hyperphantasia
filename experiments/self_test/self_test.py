"""Behavioural imagination test for any reasoner that can read text (a person, Claude, another model).

    python experiments/self_test/self_test.py make            # print puzzles, store hidden answers
    python experiments/self_test/self_test.py score FILE      # score answers (0/1 per line or space-separated)

Covers criterion 1 of the detector (docs/IMAGINATION_DETECTOR.md): questions that need premises
composed, invariance to reshuffling, and longer worlds. It can rule imagination out (a surface
heuristic scores about 57% on composition) but cannot show where imagination lives; that needs
access to the subject's internals.
"""

import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperphantasia.detect import world as W  # noqa: E402

HERE = Path(__file__).resolve().parent
L = lambda t: chr(65 + t)


def make(seed: int = 2026):
    rng = np.random.default_rng(seed)
    items = []
    while len(items) < 20:  # five-letter worlds, enriched for questions that need composition
        p = W.sample(rng, 5)
        if p.needs_composition() or rng.random() < 0.35:
            items.append(("main", p))
    items += [("reshuffle", W.reshuffle(p, rng)) for _, p in items[:8]]
    items += [("long", W.sample(rng, 7)) for _ in range(6)]
    items = [items[i] for i in rng.permutation(len(items))]
    pickle.dump(items, open(HERE / "items.pkl", "wb"))
    for k, (_, p) in enumerate(items, 1):
        prem = " ; ".join(f"{L(a)} {'<' if op == W.LT else '>'} {L(b)}" for a, op, b in p.premises)
        print(f"{k:>2}. {prem}   ?  is {L(p.query[0])} left of {L(p.query[1])}")


def score(path: str):
    items = pickle.load(open(HERE / "items.pkl", "rb"))
    mine = [int(x) for x in Path(path).read_text().split()]
    assert len(mine) == len(items), f"expected {len(items)} answers"
    ok = [m == p.answer for m, (_, p) in zip(mine, items)]

    def acc(sel):
        s = [o for o, (_, p) in zip(ok, items) if sel(p)]
        return f"{sum(s)}/{len(s)}"

    print("all items                  ", acc(lambda p: True))
    print("needing composition (n=5)  ", acc(lambda p: p.n == 5 and p.needs_composition()))
    print("stated pairs (n=5)         ", acc(lambda p: p.n == 5 and p.adjacent()))
    print("7-letter worlds            ", acc(lambda p: p.n == 7))
    pairs = {}
    for m, (_, p) in zip(mine, items):
        pairs.setdefault((p.order, p.query), []).append(m)
    same = [len(set(v)) == 1 for v in pairs.values() if len(v) == 2]
    print("same answer after reshuffle", f"{sum(same)}/{len(same)}")


if __name__ == "__main__":
    make() if sys.argv[1] == "make" else score(sys.argv[2])
