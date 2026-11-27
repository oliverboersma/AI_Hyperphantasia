"""Exploratory: are positions encoded non-linearly in the bottleneck's slots?

A linear probe found no scene in the slots of the bottleneck network trained on mixed questions. A
small MLP probe (2 hidden layers of 256) reads [input vector, which letter] -> that letter's rank.
Because an MLP can compute ranks itself from any lossless copy of the premises, it is run on:

* the trained network's slots (decoder layer 0, flattened)
* an untrained copy's slots
* a raw one-hot encoding of the premise grid (lossless, nothing computed)

The slots count as scene-like only if ranks are easier to read from them than from the raw premises.

    python experiments/detector_followups/mlp_scene_probe.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, world as W  # noqa: E402
from hyperphantasia.detect.subjects import NeuralSubject  # noqa: E402

PATHS = {"bottleneck, mixed questions": "side-door-pos-shut-continued_seed0_steps6000.pkl",
         "bottleneck, left-of only": "side-door-shut-continued_seed0_steps6000.pkl"}


def dataset(feats, probs):
    X, y, g = [], [], []
    for i, p in enumerate(probs):
        for letter, r in p.rank.items():
            X.append(np.concatenate([feats[i], np.eye(W.LETTERS, dtype=np.float32)[letter]]))
            y.append(r)
            g.append(i)
    return np.array(X, np.float32), np.array(y), np.array(g)


def mlp_probe(X, y, groups, steps=3000, seed=0):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    test = groups % 5 == 0  # hold out whole problems
    Xtr, ytr, Xte, yte = jnp.asarray(X[~test]), jnp.asarray(y[~test]), jnp.asarray(X[test]), y[test]
    k = jax.random.split(jax.random.PRNGKey(seed), 3)
    d = X.shape[1]
    p = {"w1": jax.random.normal(k[0], (d, 256)) / np.sqrt(d), "b1": jnp.zeros(256),
         "w2": jax.random.normal(k[1], (256, 256)) / 16, "b2": jnp.zeros(256),
         "w3": jax.random.normal(k[2], (256, 5)) / 16, "b3": jnp.zeros(5)}

    def f(p, x):
        h = jax.nn.relu(x @ p["w1"] + p["b1"])
        h = jax.nn.relu(h @ p["w2"] + p["b2"])
        return h @ p["w3"] + p["b3"]

    def loss(p, x, t):
        return -jnp.take_along_axis(jax.nn.log_softmax(f(p, x)), t[:, None], 1).mean() + \
            1e-4 * sum(jnp.sum(v ** 2) for kk, v in p.items() if kk.startswith("w"))

    @jax.jit
    def step(p, m, v, t, x, yy):
        g = jax.grad(loss)(p, x, yy)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        p = jax.tree_util.tree_map(lambda a, b, c: a - 1e-3 * (b / (1 - 0.9 ** t)) / (jnp.sqrt(c / (1 - 0.999 ** t)) + 1e-8), p, m, v)
        return p, m, v

    m = jax.tree_util.tree_map(jnp.zeros_like, p)
    v = jax.tree_util.tree_map(jnp.zeros_like, p)
    rng = np.random.default_rng(seed)
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(Xtr), 512)
        p, m, v = step(p, m, v, t, Xtr[idx], ytr[idx])
    return float((np.asarray(f(p, Xte)).argmax(-1) == yte).mean())


if __name__ == "__main__":
    rng = np.random.default_rng(41)
    probs = [W.sample(rng, 5) for _ in range(8000)]
    print("MLP probe for each letter's rank (chance 0.20; held-out problems)")
    for name, path in PATHS.items():
        cfg, params, _ = nets.load(ROOT / ".cache/detect" / path)
        for label, subj in (("trained slots", NeuralSubject(cfg, params, "t")),
                            ("untrained slots", NeuralSubject(cfg, nets.init(cfg, 99), "u"))):
            _, acts = subj.run(probs)
            feats = acts[(0, "scene")].reshape(len(probs), -1)
            print(f"  {name:<28} {label:<16} {mlp_probe(*dataset(feats, probs)):.3f}", flush=True)
    toks = nets._slotted(probs, nets.NetConfig(kind="plain"), True, [[] for _ in probs])
    raw = np.eye(W.BASE_VOCAB, dtype=np.float32)[toks].reshape(len(probs), -1)
    print(f"  {'raw premises (lossless)':<28} {'one-hot grid':<16} {mlp_probe(*dataset(raw, probs)):.3f}")
