"""The test bed: hidden-order worlds.

Each problem has a secret left-to-right order of N letters (the world). The
subject sees only the N-1 neighbouring pairs, shuffled, each written either
as "a < b" or as "b > a", followed by a question "is x left of y?".

The task is useful for detecting imagination because:

* answering a question about two letters that never appear in the same
  premise requires composing several premises (transitive inference);
* simple surface statistics (how often a letter appears on the small side)
  only identify the two ends of the line, so they cannot solve the middle;
* the world has a crisp state (each letter's rank) that can be looked for
  inside a network and edited through counterfactual inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

LETTERS = 26
LT, GT, QMARK, ANS, BOS, PAD = 26, 27, 28, 29, 30, 31
BASE_VOCAB = 32
MAX_N = 7
MAX_PREMISE_TOKENS = 3 * (MAX_N - 1)


@dataclass(frozen=True)
class Problem:
    order: tuple[int, ...]                       # letters, left to right (the hidden world)
    premises: tuple[tuple[int, int, int], ...]   # (token, op, token) as presented
    query: tuple[int, int]

    @property
    def rank(self) -> dict[int, int]:
        return {letter: i for i, letter in enumerate(self.order)}

    @property
    def answer(self) -> int:
        r = self.rank
        return int(r[self.query[0]] < r[self.query[1]])

    @property
    def n(self) -> int:
        return len(self.order)

    def adjacent(self) -> bool:
        r = self.rank
        return abs(r[self.query[0]] - r[self.query[1]]) == 1

    def needs_composition(self) -> bool:
        """Neither letter is an end and the pair is not stated in a premise."""
        r = self.rank
        ends = {0, self.n - 1}
        return not self.adjacent() and r[self.query[0]] not in ends and r[self.query[1]] not in ends


def premises_for(order, rng) -> tuple[tuple[int, int, int], ...]:
    pairs = [(order[i], order[i + 1]) for i in range(len(order) - 1)]
    rng.shuffle(pairs)
    out = []
    for a, b in pairs:  # a is left of b
        out.append((a, LT, b) if rng.random() < 0.5 else (b, GT, a))
    return tuple(out)


def sample(rng: np.random.Generator, n: int) -> Problem:
    order = tuple(int(x) for x in rng.choice(LETTERS, size=n, replace=False))
    x, y = (int(v) for v in rng.choice(order, size=2, replace=False))
    return Problem(order, premises_for(order, rng), (x, y))



def batch(rng, size: int, ns=(3, 4, 5, 6)) -> list[Problem]:
    return [sample(rng, int(rng.choice(ns))) for _ in range(size)]


def reshuffle(p: Problem, rng) -> Problem:
    """Same world and question, different presentation."""
    return replace(p, premises=premises_for(p.order, rng))


def requery(p: Problem, q: tuple[int, int]) -> Problem:
    return replace(p, query=q)


def scene_counterfactual(p: Problem, rng, tries: int = 50) -> Problem | None:
    """Same letters and question, a different world in which the answer flips."""
    for _ in range(tries):
        order = tuple(int(v) for v in rng.permutation(p.order))
        cf = Problem(order, premises_for(order, rng), p.query)
        if cf.answer != p.answer:
            return cf
    return None


def rank_counterfactual(p: Problem, rng) -> Problem | None:
    """Move only x: swap it with a letter on the far side of y.

    Then y keeps its rank, x's rank changes, and the answer flips. If a network
    answers by comparing internal 'rank of x' and 'rank of y' variables, then
    transplanting just the 'rank of x' variable from this world must flip it.
    """
    x, y = p.query
    r = p.rank
    side = [z for z in p.order if z not in (x, y) and (r[z] > r[y]) == (r[x] < r[y])]
    if not side:
        return None
    z = int(rng.choice(side))
    order = list(p.order)
    order[r[x]], order[r[z]] = z, x
    cf = Problem(tuple(order), premises_for(tuple(order), rng), p.query)
    assert cf.rank[y] == r[y] and cf.answer != p.answer
    return cf


# ------------------------------------------------------------------ encodings
def encode_sequence(problems: list[Problem], length: int = 1 + MAX_PREMISE_TOKENS + 4) -> np.ndarray:
    """[PAD.. BOS premises ? x y =] left-padded so the question block is always last."""
    out = np.full((len(problems), length), PAD, dtype=np.int32)
    for i, p in enumerate(problems):
        toks = [BOS] + [t for prem in p.premises for t in prem] + [QMARK, p.query[0], p.query[1], ANS]
        out[i, length - len(toks):] = toks
    return out


def encode_premises(problems: list[Problem], length: int = 1 + MAX_PREMISE_TOKENS) -> np.ndarray:
    out = np.full((len(problems), length), PAD, dtype=np.int32)
    for i, p in enumerate(problems):
        toks = [BOS] + [t for prem in p.premises for t in prem]
        out[i, length - len(toks):] = toks
    return out


def labels(problems: list[Problem]) -> np.ndarray:
    return np.array([p.answer for p in problems], dtype=np.int32)


# ----------------------------------------------------------- surface features
def surface_features(p: Problem) -> dict[int, np.ndarray]:
    """What a reader could know about each letter without composing premises."""
    feats = {}
    first = {}
    for k, (a, op, b) in enumerate(p.premises):
        for t in (a, b):
            first.setdefault(t, k)
    for letter in p.order:
        small = sum(1 for a, op, b in p.premises if (a == letter and op == LT) or (b == letter and op == GT))
        big = sum(1 for a, op, b in p.premises if (a == letter and op == GT) or (b == letter and op == LT))
        feats[letter] = np.array([small, big, first[letter], small - big], dtype=np.float32)
    return feats
