"""Exploratory follow-up (post hoc): is the plain transformer's x-site code a membership list?

The detector judged the trained plain transformer "something else": x's site at layer 2 is
causally important (transplanting it moves the answer 81% of the time), but rank is not
linearly decodable there and its rank subspace has no effect. This script asks whether the
site instead encodes "which letters are left of x", and whether editing one letter's
direction flips the answer. Run after `python -m hyperphantasia.detect` has trained the net.
Output from our run: relational_code.out.txt
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperphantasia.detect import nets, probes, world as W
from hyperphantasia.detect.subjects import NeuralSubject, Patch

cfg, params, _ = nets.load(Path(__file__).resolve().parents[2] / ".cache/detect/plain_seed0_steps8000_curriculum.pkl")
net = NeuralSubject(cfg, params, "plain")
ctl = NeuralSubject(cfg, nets.init(cfg, 99), "untrained")
rng = np.random.default_rng(21)
probs = [W.sample(rng, 5) for _ in range(8000)]
_, acts = net.run(probs)
_, cacts = ctl.run(probs)

def surface(p, L, x):
    f = W.surface_features(p)
    direct_lx = any((a == L and op == W.LT and b == x) or (a == x and op == W.GT and b == L) for a, op, b in p.premises)
    direct_xl = any((a == x and op == W.LT and b == L) or (a == L and op == W.GT and b == x) for a, op, b in p.premises)
    return np.concatenate([f[x][:2], f[L][:2], [direct_lx, direct_xl]]).astype(np.float32)

def relational_probes(A, site_letter):
    """One probe per letter L: is L left of the site's letter?"""
    out, fitted = [], {}
    for L in range(W.LETTERS):
        idx = [i for i, p in enumerate(probs) if L in p.rank and site_letter(p) != L]
        if len(idx) < 200:
            continue
        y = np.array([int(probs[i].rank[L] < probs[i].rank[site_letter(probs[i])]) for i in idx])
        pr = probes.fit(A[idx].reshape(len(idx), -1), y, 2, seed=L)
        fitted[L] = pr
        out.append((pr.acc, len(idx)))
    return np.average([a for a, _ in out], weights=[n for _, n in out]), fitted

print("Probe: 'is letter L left of x?' at x's site (chance 0.50)")
for layer in (1, 2, 3):
    acc, _ = relational_probes(acts[(layer, "X")], lambda p: p.query[0])
    cacc, _ = relational_probes(cacts[(layer, "X")], lambda p: p.query[0])
    print(f"  layer {layer}: trained {acc:.3f}   untrained {cacc:.3f}")
# surface baseline: same question from local, non-composed features
scores = []
for L in range(W.LETTERS):
    idx = [i for i, p in enumerate(probs) if L in p.rank and p.query[0] != L]
    X = np.array([surface(probs[i], L, probs[i].query[0]) for i in idx])
    y = np.array([int(probs[i].rank[L] < probs[i].rank[probs[i].query[0]]) for i in idx])
    scores.append((probes.fit(X, y, 2, seed=L).acc, len(idx)))
print(f"  surface counts + stated pairs: {np.average([a for a, _ in scores], weights=[n for _, n in scores]):.3f}")

layer = 2
_, fitted = relational_probes(acts[(layer, "X")], lambda p: p.query[0])

# Causal test: move only x (y keeps its rank); at x's site, layer 2, transplant ONE direction.
base, cf = [], []
while len(base) < 1000:
    p = W.sample(rng, 5)
    c = W.rank_counterfactual(p, rng)
    if c is not None:
        base.append(p); cf.append(c)
_, cf_acts = net.run(cf)
target = W.labels(cf)
vals = cf_acts[(layer, "X")]
d = vals.shape[-1]

def direction(pr):
    w = pr.W[:, 1] - pr.W[:, 0]
    return w / np.linalg.norm(w)

def patched(per_item_dirs):
    """Transplant, item by item, only the component along the given unit vector(s)."""
    out = np.empty(len(base), dtype=int)
    groups = {}
    for i, U in enumerate(per_item_dirs):
        groups.setdefault(U.tobytes(), (U, []))[1].append(i)
    for U, idx in groups.values():
        P = U @ U.T
        pred, _ = net.run([base[i] for i in idx], Patch(layer, "X", vals[idx], P))
        out[idx] = pred
    return (out == target).mean()

y_dir = [direction(fitted[p.query[1]])[:, None] for p in base]
other = []
for p in base:
    choices = [L for L in p.order if L not in p.query and L in fitted]
    other.append(direction(fitted[choices[0]])[:, None])
rand = [probes.random_subspace(d, 1, np.random.default_rng(i)) for i in range(len(base))]
all_dirs = np.stack([direction(fitted[L]) for L in sorted(fitted)], axis=1)
Q, _ = np.linalg.qr(all_dirs)
R = probes.random_subspace(d, Q.shape[1], np.random.default_rng(7))
print(f"\nMove only x (y keeps its rank). Transplant at x's site, layer {layer}:")
pred, _ = net.run(base, Patch(layer, "X", vals))
print(f"  everything                                     {(pred == target).mean():.2f}")
print(f"  ONE direction: 'is y left of x?'               {patched(y_dir):.2f}")
print(f"  one direction for an uninvolved letter         {patched(other):.2f}")
print(f"  one random direction                           {patched(rand):.2f}")
pred, _ = net.run(base, Patch(layer, "X", vals, Q @ Q.T))
print(f"  all {Q.shape[1]} membership directions                  {(pred == target).mean():.2f}")
pred, _ = net.run(base, Patch(layer, "X", vals, R @ R.T))
print(f"  random subspace of the same size              {(pred == target).mean():.2f}")
