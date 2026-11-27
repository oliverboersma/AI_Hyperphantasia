"""Distributed alignment search on the bottleneck's slots (docs/NEXT_STEPS.md, item 1b follow-up).

The network (Slot Attention bottleneck, trained on mixed left-of and position questions, side door
shut) is frozen. For k in {8, 32, 128}, learn an orthonormal k-dimensional subspace U of the
flattened slot space (6 x 64 = 384 dims) such that transplanting only that subspace from a source
world (same letters, a different order) into a base run makes the network answer as the source world
would. Questions are a mix of "is x left of y?" and "is x at position k?". Interchange-intervention
accuracy (IIA) is measured on held-out pairs whose base and source answers differ, against random
subspaces of the same size and the full transplant.

Then: are letter positions readable from the learned subspace, or from its complement? MLP rank
probes as in mlp_scene_probe.py. If positions live in the learned subspace, the causal part of the
slots is the scene.

    python experiments/detector_followups/das_slots.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, world as W  # noqa: E402
from mlp_scene_probe import dataset, mlp_probe  # noqa: E402

PATH = ROOT / ".cache/detect/side-door-pos-shut-continued_seed0_steps6000.pkl"
K, D = 6, 64
DIM = K * D
EYE = jnp.eye(D)

cfg, params, _ = nets.load(PATH)


def make_batch(rng, size):
    """Base worlds, source worlds (same letters, new order), and one question each."""
    base, src, qs = [], [], []
    for _ in range(size):
        p = W.sample(rng, 5)
        order = tuple(int(v) for v in rng.permutation(p.order))
        s = W.Problem(order, W.premises_for(order, rng), p.query)
        x, y = (int(v) for v in rng.choice(p.order, size=2, replace=False))
        if rng.random() < 0.5:
            q = (x, y)
            target = int(s.rank[x] < s.rank[y])
            base_ans = int(p.rank[x] < p.rank[y])
        else:
            k = s.rank[x] if rng.random() < 0.5 else int(rng.choice([r for r in range(5) if r != s.rank[x]]))
            q = ("pos", x, k)
            target = int(s.rank[x] == k)
            base_ans = int(p.rank[x] == k)
        base.append(p); src.append(s); qs.append((q, target, base_ans))
    return base, src, qs


def tokens(probs, qs, rng):
    dummy = [tuple(int(v) for v in rng.choice(p.order, size=2, replace=False)) for p in probs]
    return jnp.asarray(np.concatenate([nets._slotted(probs, cfg, True, [[] for _ in probs]),
                                       nets._slotted(probs, cfg, False, [[d, q] for d, q in zip(dummy, qs)])],
                                      axis=1))


def slots(probs):
    _, resid = nets.forward(params, jnp.asarray(nets.encode(cfg, probs)), cfg)
    return resid[0][:, :K].reshape(len(probs), DIM)


def frame(Wm):
    q, _ = jnp.linalg.qr(Wm)
    return q  # [DIM, k], orthonormal columns


def run(toks, hb, hs, U):
    patched = hb + ((hs - hb) @ U) @ U.T
    logits, _ = nets.forward(params, toks, cfg, 0, jnp.arange(K), patched.reshape(-1, K, D), EYE)
    return logits


def loss_fn(Wm, toks, hb, hs, y):
    logits = run(toks, hb, hs, frame(Wm))
    return -jnp.take_along_axis(jax.nn.log_softmax(logits), y[:, None], 1).mean()


grad_fn = jax.jit(jax.value_and_grad(loss_fn))


def train_subspace(k, steps=1000, seed=0):
    rng = np.random.default_rng(seed)
    Wm = jnp.asarray(np.random.default_rng(seed).normal(size=(DIM, k)) / np.sqrt(DIM))
    m, v = jnp.zeros_like(Wm), jnp.zeros_like(Wm)
    for t in range(1, steps + 1):
        base, src, qs = make_batch(rng, 128)
        toks = tokens(base, [q for q, _, _ in qs], rng)
        y = jnp.asarray([tgt for _, tgt, _ in qs])
        loss, g = grad_fn(Wm, toks, slots(base), slots(src), y)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        Wm = Wm - 1e-2 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
        if t % 250 == 0:
            print(f"    k={k} step {t} loss {float(loss):.3f}", flush=True)
    return frame(Wm)


def iia(U, n=2000, seed=99):
    rng = np.random.default_rng(seed)
    base, src, qs = make_batch(rng, n)
    toks = tokens(base, [q for q, _, _ in qs], rng)
    pred = np.asarray(run(toks, slots(base), slots(src), U)).argmax(-1)
    tgt = np.array([t for _, t, _ in qs])
    differs = np.array([t != b for _, t, b in qs])
    return float((pred == tgt)[differs].mean())


if __name__ == "__main__":
    print("Interchange-intervention accuracy on held-out pairs whose answers differ (chance-level disruption ~0.5)")
    print(f"  full transplant of the slots: {iia(jnp.eye(DIM)):.3f}")
    learned = {}
    for k in (8, 32, 128):
        U = train_subspace(k)
        learned[k] = U
        R = frame(jnp.asarray(np.random.default_rng(k).normal(size=(DIM, k))))
        print(f"  k={k:<4} learned subspace {iia(U):.3f}   random subspace {iia(R):.3f}", flush=True)

    print("\nWhere are letter positions? MLP rank probe (raw premises reach 0.35; all slots 0.83)")
    rng = np.random.default_rng(41)
    probs = [W.sample(rng, 5) for _ in range(8000)]
    S = np.asarray(slots(probs))
    for k in (8, 32):
        U = np.asarray(learned[k])
        inside = S @ U
        outside = S - inside @ U.T
        print(f"  k={k:<4} from the learned subspace only {mlp_probe(*dataset(inside, probs)):.3f}   "
              f"from everything else {mlp_probe(*dataset(outside, probs)):.3f}", flush=True)
