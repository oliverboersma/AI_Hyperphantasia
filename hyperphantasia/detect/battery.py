"""The battery: behaviour, representation, causation, timing.

1. Behaviour       Does it answer questions that need premises composed? Is it
                   invariant to how the premises are shuffled and phrased?
2. Representation  Can each letter's rank be read linearly from the
                   activations, better than from an untrained copy of the same
                   network and better than from surface counts alone?
3. Causation       Transplant activations from a world where the answer is
                   different. Does the answer follow the transplanted world?
                   Transplant *only* the decoded rank subspace: does that
                   suffice, compared with a random subspace of the same size?
4. Timing          Does a whole-scene representation exist at the "scene" site
                   (after the premises, before the question), or only at the
                   positions of the letters asked about?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import probes
from . import world as W
from .subjects import Patch, Subject

N_EVAL = 5  # evaluate on 5-letter worlds: ranks 0..4


@dataclass
class Evidence:
    subject: str
    behaviour: dict = field(default_factory=dict)
    probes: dict = field(default_factory=dict)      # (layer, role) -> {"acc", "untrained", "surface", "chance"}
    tracing: dict = field(default_factory=dict)     # (layer, role) -> transfer rate under a scene counterfactual
    subspace: dict = field(default_factory=dict)    # layer -> {"full", "rank", "random", "dims"} at role X


def _problems(seed, n, count):
    rng = np.random.default_rng(seed)
    return [W.sample(rng, n) for _ in range(count)]


# ---------------------------------------------------------------- behaviour
def behaviour(subject: Subject, seed: int = 11, count: int = 3000) -> dict:
    probs = _problems(seed, N_EVAL, count)
    pred, _ = subject.run(probs)
    truth = W.labels(probs)
    ok = pred == truth
    comp = np.array([p.needs_composition() for p in probs])
    adj = np.array([p.adjacent() for p in probs])
    rng = np.random.default_rng(seed + 1)
    base = probs[:600]
    variants = [[W.reshuffle(p, rng) for p in base] for _ in range(4)]
    outs = np.stack([subject.run(v)[0] for v in variants])
    long = _problems(seed + 2, W.MAX_N, 1000)
    return {
        "accuracy": float(ok.mean()),
        "composition": float(ok[comp].mean()),
        "stated_pairs": float(ok[adj].mean()),
        "shuffle_consistency": float((outs == outs[0]).all(axis=0).mean()),
        "longer_worlds": float((subject.run(long)[0] == W.labels(long)).mean()),
    }


# ----------------------------------------------------------- representation
def _rank_targets(probs, role):
    if role == "X":
        return np.array([p.rank[p.query[0]] for p in probs])
    if role == "Y":
        return np.array([p.rank[p.query[1]] for p in probs])
    raise ValueError(role)


def _surface(probs, role):
    rows = []
    for p in probs:
        letter = p.query[0] if role == "X" else p.query[1]
        rows.append(W.surface_features(p)[letter])
    return np.array(rows)


def _scene_probe(acts, probs, seed):
    """Per-letter probes reading every letter's rank from the scene site."""
    scores, weights = [], []
    for letter in range(W.LETTERS):
        idx = [i for i, p in enumerate(probs) if letter in p.rank]
        if len(idx) < 100:
            continue
        y = np.array([probs[i].rank[letter] for i in idx])
        pr = probes.fit(acts[idx], y, N_EVAL, seed)
        scores.append(pr.acc)
        weights.append(len(idx))
    return float(np.average(scores, weights=weights))


def _scene_surface(probs, seed):
    scores, weights = [], []
    for letter in range(W.LETTERS):
        idx = [i for i, p in enumerate(probs) if letter in p.rank]
        if len(idx) < 100:
            continue
        X = np.array([W.surface_features(probs[i])[letter] for i in idx])
        y = np.array([probs[i].rank[letter] for i in idx])
        scores.append(probes.fit(X, y, N_EVAL, seed).acc)
        weights.append(len(idx))
    return float(np.average(scores, weights=weights))


def representation(subject: Subject, control: Subject | None, seed: int = 21, count: int = 6000):
    probs = _problems(seed, N_EVAL, count)
    _, acts = subject.run(probs)
    _, cacts = control.run(probs) if control else (None, {})
    out, fitted = {}, {}
    surface_xy = {r: probes.fit(_surface(probs, r), _rank_targets(probs, r), N_EVAL, seed).acc for r in ("X", "Y")}
    surface_scene = _scene_surface(probs, seed)
    for (layer, role), a in acts.items():
        if role == "ANS":
            continue
        if role == "scene":
            acc = _scene_probe(a.reshape(len(a), -1), probs, seed)
            ctrl = _scene_probe(cacts[(layer, role)].reshape(len(a), -1), probs, seed) if control else None
            surf = surface_scene
        else:
            y = _rank_targets(probs, role)
            pr = probes.fit(a.reshape(len(a), -1), y, N_EVAL, seed)
            fitted[(layer, role)] = pr
            acc = pr.acc
            ctrl = probes.fit(cacts[(layer, role)].reshape(len(a), -1), y, N_EVAL, seed).acc if control else None
            surf = surface_xy[role]
        out[(layer, role)] = {"acc": acc, "untrained": ctrl, "surface": surf, "chance": 1 / N_EVAL}
    return out, fitted


# ----------------------------------------------------------------- causation
def _pairs(seed, count, make_cf):
    rng = np.random.default_rng(seed)
    base, cf = [], []
    while len(base) < count:
        p = W.sample(rng, N_EVAL)
        c = make_cf(p, rng)
        if c is not None:
            base.append(p)
            cf.append(c)
    return base, cf


def tracing(subject: Subject, seed: int = 31, count: int = 800) -> dict:
    """Interchange intervention: transplant a whole site from a world with the other answer."""
    base, cf = _pairs(seed, count, W.scene_counterfactual)
    _, cacts = subject.run(cf)
    target = W.labels(cf)
    out = {}
    for (layer, role), vals in cacts.items():
        pred, _ = subject.run(base, Patch(layer, role, vals))
        out[(layer, role)] = float((pred == target).mean())
    return out


def subspace(subject: Subject, fitted: dict, seed: int = 41, count: int = 800) -> dict:
    """Move only x (y keeps its rank). Transplant all of x's site, only its rank subspace, or a random one."""
    base, cf = _pairs(seed, count, W.rank_counterfactual)
    _, cacts = subject.run(cf)
    target = W.labels(cf)
    rng = np.random.default_rng(seed)
    out = {}
    for layer in subject.layers:
        key = (layer, "X")
        if key not in cacts or key not in fitted:
            continue
        vals = cacts[key]
        d = vals.shape[-1]
        U = fitted[key].subspace()
        R = probes.random_subspace(d, U.shape[1], rng)
        res = {"dims": int(U.shape[1])}
        for name, proj in (("full", None), ("rank", U @ U.T), ("random", R @ R.T)):
            pred, _ = subject.run(base, Patch(layer, "X", vals, proj))
            res[name] = float((pred == target).mean())
        out[layer] = res
    return out


def examine(subject: Subject, control: Subject | None = None, quick: bool = False) -> Evidence:
    scale = 0.25 if quick else 1.0
    ev = Evidence(subject.name)
    ev.behaviour = behaviour(subject, count=int(3000 * scale))
    ev.probes, fitted = representation(subject, control, count=int(6000 * scale))
    ev.tracing = tracing(subject, count=int(800 * scale))
    ev.subspace = subspace(subject, fitted, count=int(800 * scale))
    return ev
