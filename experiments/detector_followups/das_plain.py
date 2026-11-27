"""Distributed alignment search on the plain transformer's on-demand code (docs/NEXT_STEPS.md, item 4).

The plain transformer forms no standing scene. When the question "is x left of y?" names x, it
builds a code at x's site (layer 2) that carries the answer (transplanting the site moves it 81% of
the time), yet rank is not linearly readable there (relational_code.py). This script learns, on the
frozen network, the subspace of that site the answer actually depends on, and then asks what
variable it holds.

A. Size. Train on "move only x" counterfactuals (x swaps with a letter beyond y; y keeps its rank).
   Interchange accuracy of learned k-dim subspaces against random ones and the full 64-dim site.

B. Coordinate or relation list? Source = the same letters in a new order. Transplanting x's code
   from it can mean two different answers:
     coordinate: x's site carries x's rank, compared later with y's rank in the base world
                 -> answer rank_src(x) < rank_base(y)
     relations:  x's site carries where the other letters lie relative to x, in the source world
                 -> answer rank_src(x) < rank_src(y), the source world's own answer
   Each is scored on the pairs where its answer differs from the base answer (on pairs where only
   the two hypotheses differ, the base answer already agrees with "coordinate" 86% of the time, so
   that comparison says nothing; see das_plain_coordinate.py). DAS is also fitted to each
   hypothesis's targets; the better-aligned causal model reaches the higher interchange accuracy.

C. What the learned subspace holds: linear readouts of x's rank and of "is L left of x?".

    python experiments/detector_followups/das_plain.py [checkpoint]
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, probes, world as W  # noqa: E402

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache/detect/plain_seed0_steps8000_curriculum.pkl"
LAYER = 2
cfg, params, _ = nets.load(PATH)
POS = nets.roles(cfg)["X"][0]
D = cfg.d
EYE = jnp.eye(D)


def site(probs):
    toks = jnp.asarray(nets.encode(cfg, probs))
    _, resid = nets.forward(params, toks, cfg)
    return toks, resid[LAYER][:, POS]


def run(toks, hb, hs, U):
    patched = hb + ((hs - hb) @ U) @ U.T
    logits, _ = nets.forward(params, toks, cfg, LAYER, jnp.asarray([POS]), patched[:, None, :], EYE)
    return logits


def frame(Wm):
    q, _ = jnp.linalg.qr(Wm)
    return q


def loss_fn(Wm, toks, hb, hs, y, w):
    logits = run(toks, hb, hs, frame(Wm))
    nll = -jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], 1)[:, 0]
    return (nll * w).sum() / w.sum()


grad_fn = jax.jit(jax.value_and_grad(loss_fn))


def pairs(rng, size, kind):
    """(base, source, targets) where targets = {name: (answer, valid)}."""
    base, src = [], []
    while len(base) < size:
        p = W.sample(rng, 5)
        if kind == "move_x":
            s = W.rank_counterfactual(p, rng)
            if s is None:
                continue
        else:
            order = tuple(int(v) for v in rng.permutation(p.order))
            s = W.Problem(order, W.premises_for(order, rng), p.query)
        base.append(p); src.append(s)
    x = [p.query[0] for p in base]; y = [p.query[1] for p in base]
    t = {
        "relations": (np.array([s.answer for s in src]), np.ones(size, bool)),
        "coordinate": (np.array([int(s.rank[a] < p.rank[b]) for p, s, a, b in zip(base, src, x, y)]),
                       np.array([s.rank[a] != p.rank[b] for p, s, a, b in zip(base, src, x, y)])),
        "base": (np.array([p.answer for p in base]), np.ones(size, bool)),
    }
    return base, src, t


def train(k, kind, hyp, steps=800, seed=0):
    rng = np.random.default_rng(seed)
    Wm = jnp.asarray(np.random.default_rng(seed + 1).normal(size=(D, k)) / np.sqrt(D))
    m, v = jnp.zeros_like(Wm), jnp.zeros_like(Wm)
    for t in range(1, steps + 1):
        base, src, tg = pairs(rng, 256, kind)
        ans, ok = tg[hyp]
        toks, hb = site(base)
        _, hs = site(src)
        _, g = grad_fn(Wm, toks, hb, hs, jnp.asarray(ans), jnp.asarray(ok, jnp.float32))
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        Wm = Wm - 1e-2 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return frame(Wm)


EVAL = {kind: pairs(np.random.default_rng(99 + i), 3000, kind) for i, kind in enumerate(("move_x", "reorder"))}
CACHE = {}


def predict(kind, U):
    base, src, _ = EVAL[kind]
    if kind not in CACHE:
        toks, hb = site(base)
        CACHE[kind] = (toks, hb, site(src)[1])
    toks, hb, hs = CACHE[kind]
    return np.asarray(run(toks, hb, hs, U)).argmax(-1)


def iia(kind, U, hyp):
    """Accuracy against hyp's answer, on valid pairs where it differs from the base answer."""
    _, _, tg = EVAL[kind]
    ans, ok = tg[hyp]
    keep = ok & (ans != tg["base"][0])
    return float((predict(kind, U) == ans)[keep].mean())


def rand(k, seed):
    return frame(jnp.asarray(np.random.default_rng(seed).normal(size=(D, k))))


if __name__ == "__main__":
    print(f"{PATH.name}, x's site, layer {LAYER}")
    print("\nA. Move only x (held-out pairs whose answers differ). Answer follows the source:")
    print(f"  whole site (64 dims)   {iia('move_x', EYE, 'relations'):.3f}")
    learned = {}
    for k in (1, 2, 4, 8, 16):
        learned[k] = train(k, "move_x", "relations")
        print(f"  k={k:<3} learned {iia('move_x', learned[k], 'relations'):.3f}   "
              f"random {iia('move_x', rand(k, k), 'relations'):.3f}", flush=True)

    print("\nB. New order of the same letters. Interchange accuracy under each hypothesis, each scored on")
    print("   the pairs where that hypothesis's answer differs from the base answer:")
    print(f"  {'':<28} {'relations':>10} {'coordinate':>11}")
    for name, U in [("whole site", EYE), ("learned on move-x, k=1", learned[1]),
                    ("learned on move-x, k=8", learned[8])]:
        print(f"  {name:<28} {iia('reorder', U, 'relations'):>10.3f} {iia('reorder', U, 'coordinate'):>11.3f}")
    fit = {hyp: train(8, "reorder", hyp, seed=5) for hyp in ("relations", "coordinate")}
    print(f"  {'DAS fitted to that hypothesis, k=8':<28} {iia('reorder', fit['relations'], 'relations'):>10.3f} "
          f"{iia('reorder', fit['coordinate'], 'coordinate'):>11.3f}", flush=True)

    print("\nC. Linear readouts at x's site (held-out accuracy)")
    rng = np.random.default_rng(41)
    probs = [W.sample(rng, 5) for _ in range(8000)]
    S = np.asarray(site(probs)[1])
    xr = np.array([p.rank[p.query[0]] for p in probs])

    def membership(A):
        scores = []
        for L in range(W.LETTERS):
            idx = [i for i, p in enumerate(probs) if L in p.rank and p.query[0] != L]
            if len(idx) < 200:
                continue
            yv = np.array([int(probs[i].rank[L] < probs[i].rank[probs[i].query[0]]) for i in idx])
            scores.append((probes.fit(A[idx], yv, 2, seed=L).acc, len(idx)))
        return np.average([a for a, _ in scores], weights=[n for _, n in scores])

    for name, A in [("whole site (64)", S), ("learned k=4", S @ np.asarray(learned[4])),
                    ("learned k=8", S @ np.asarray(learned[8])), ("random k=8", S @ np.asarray(rand(8, 8)))]:
        print(f"  {name:<16} rank of x (chance 0.20) {probes.fit(A, xr, 5).acc:.3f}   "
              f"'is L left of x?' (chance 0.50) {membership(A):.3f}", flush=True)
