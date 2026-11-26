"""Plato's Key for imagination: can one network use another's on-demand code?

Adapted from Hazy Research's "Plato's Key" (2026): a single rotation and one global scale,
fit in closed form, let independently trained models use each other's hidden states. Here
the hidden state is the code that the plain transformer builds at the asked-about letter
(x's site), which the detector found to be causally used but not a coordinate.

For every ordered pair of independently trained plain transformers (different seeds):

1. Fit z_B = mu_B + s R (z_A - mu_A) on x-site activations from 3000 shared puzzles, with R
   the orthogonal Procrustes rotation and s = ||Z_B|| / ||Z_A||.
2. Take puzzles and their rank counterfactuals (only x moves, y keeps its place, the answer
   flips). Run the donor A on the counterfactual, map its x-site state into B, and swap it into
   B's run on the original puzzle.
3. Transfer = how often B's answer follows the counterfactual world.

Reference: self-swap (B's own counterfactual state). Controls: a random rotation with the same
scale, and a rotation fit on mismatched puzzles. Layers are chosen on a validation split and
reported on a test split. Finally a chain A -> B -> C is compared with the direct A -> C map.

    python experiments/detector_followups/rotation_transfer.py
"""

import sys
from itertools import permutations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, world as W  # noqa: E402
from hyperphantasia.detect.subjects import NeuralSubject, Patch  # noqa: E402

LAYERS = (1, 2, 3, 4)


def load(seed):
    path = ROOT / f".cache/detect/plain_seed{seed}_steps8000_curriculum.pkl"
    if not path.exists():
        return None
    cfg, params, _ = nets.load(path)
    return NeuralSubject(cfg, params, f"seed {seed}")


def procrustes(ZA, ZB):
    """Columns are samples. Returns (mu_A, mu_B, s, R) with z_B ~ mu_B + s R (z_A - mu_A)."""
    muA, muB = ZA.mean(1, keepdims=True), ZB.mean(1, keepdims=True)
    A, B = ZA - muA, ZB - muB
    U, _, Vt = np.linalg.svd(B @ A.T)
    return muA, muB, np.linalg.norm(B) / np.linalg.norm(A), U @ Vt


def apply(m, Z):
    muA, muB, s, R = m
    return muB + s * R @ (Z - muA)


def compose(m1, m2):
    """m2 after m1, without running the middle network."""
    muA, muB1, s1, R1 = m1
    muB2, muC, s2, R2 = m2
    return muA, muC + s2 * R2 @ (muB1 - muB2), s1 * s2, R2 @ R1


def site(acts, layer):
    return acts[(layer, "X")][:, 0, :].T  # [d, n]


def transfer(recipient, base, target, layer_b, states):
    """Swap per-puzzle states [d, n] into the recipient's x site and score against target."""
    pred, _ = recipient.run(base, Patch(layer_b, "X", states.T[:, None, :]))
    return float((pred == target).mean())


def main():
    nets_by_seed = {s: n for s in (0, 1, 2) if (n := load(s)) is not None}
    if len(nets_by_seed) < 2:
        print("need at least two trained plain transformers (seeds 0, 1, 2)")
        return
    rng = np.random.default_rng(5)
    fit = [W.sample(rng, 5) for _ in range(3000)]
    base, cf = [], []
    while len(base) < 1600:
        p = W.sample(rng, 5)
        c = W.rank_counterfactual(p, rng)
        if c is not None:
            base.append(p), cf.append(c)
    val, test = slice(0, 600), slice(600, 1600)
    target = W.labels(cf)
    fit_acts = {s: n.run(fit)[1] for s, n in nets_by_seed.items()}
    cf_acts = {s: n.run(cf)[1] for s, n in nets_by_seed.items()}

    def score(a, b, la, lb, which, model=None):
        rec = nets_by_seed[b]
        part_base, part_t = base[which], target[which]
        if model is None:  # self-swap: the recipient's own counterfactual state
            states = site(cf_acts[b], lb)[:, which]
        else:
            states = apply(model, site(cf_acts[a], la)[:, which])
        return transfer(rec, part_base, part_t, lb, states)

    print(f"networks: {sorted(nets_by_seed)}   (x's site; transfer = answer follows the moved x)\n")
    print(f"{'pair':<8} {'layers':<8} {'rotation':>9} {'self-swap':>10} {'kept':>6} "
          f"{'random rot':>11} {'mismatched':>11}")
    best_layers = {}
    for a, b in permutations(sorted(nets_by_seed), 2):
        best = None
        for la in LAYERS:
            for lb in LAYERS:
                m = procrustes(site(fit_acts[a], la), site(fit_acts[b], lb))
                v = score(a, b, la, lb, val, m)
                if best is None or v > best[0]:
                    best = (v, la, lb, m)
        _, la, lb, m = best
        best_layers[(a, b)] = (la, lb)
        rot = score(a, b, la, lb, test, m)
        own = score(a, b, la, lb, test)
        d = m[3].shape[0]
        Q, _ = np.linalg.qr(np.random.default_rng(1).normal(size=(d, d)))
        rand = score(a, b, la, lb, test, (m[0], m[1], m[2], Q))
        perm = np.random.default_rng(2).permutation(len(fit))
        mism = score(a, b, la, lb, test, procrustes(site(fit_acts[a], la), site(fit_acts[b], lb)[:, perm]))
        print(f"{a}->{b:<5} {la},{lb:<6} {rot:>9.2f} {own:>10.2f} {rot / max(own, 1e-9):>6.0%} "
              f"{rand:>11.2f} {mism:>11.2f}")

    if len(nets_by_seed) >= 3:
        print("\nChaining at layer 2 everywhere (A -> B -> C without running B, against A -> C direct):")
        for a, b, c in permutations(sorted(nets_by_seed), 3):
            fa, fb, fc = (site(fit_acts[s], 2) for s in (a, b, c))
            chain = compose(procrustes(fa, fb), procrustes(fb, fc))
            direct = procrustes(fa, fc)
            ch = score(a, c, 2, 2, test, chain)
            di = score(a, c, 2, 2, test, direct)
            print(f"  {a}->{b}->{c}: chain {ch:.2f}   direct {a}->{c}: {di:.2f}   kept {ch / max(di, 1e-9):.0%}")


if __name__ == "__main__":
    main()
