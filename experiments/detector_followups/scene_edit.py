"""Causal test of the non-linear scene in the bottleneck's slots (docs/NEXT_STEPS.md, item 1b).

An MLP probe reads each letter's rank from the slots (0.83 for the bottleneck trained on mixed
questions). Here the slots are edited, by gradient steps on the probe, until the probe reads a
counterfactual scene: x swapped with a letter z on the far side of y, all other letters kept. The
network then runs on the edited slots. If it computes from that scene:

* flip:        "is x left of y?" follows the edited scene (the answer flips)
* specificity: a question about two letters the edit did not move keeps its answer
* position:    "is x at its new position?" becomes yes

Controls: the same editing with a probe trained on shuffled labels (it reads nothing), a random
edit of the same size, and the ceiling of transplanting the real slots of the counterfactual world.

    python experiments/detector_followups/scene_edit.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, world as W  # noqa: E402

NETS = {"bottleneck, mixed questions": "side-door-pos-shut-continued_seed0_steps6000.pkl",
        "bottleneck, left-of only": "side-door-shut-continued_seed0_steps6000.pkl"}
K, D = 6, 64
EYE = jnp.eye(D)


def slots_of(params, cfg, probs):
    toks = jnp.asarray(nets.encode(cfg, probs))
    _, resid = nets.forward(params, toks, cfg)
    return np.asarray(resid[0][:, :K])                     # decoder layer 0 = the slots


def answer(params, cfg, probs, questions, slots=None):
    """Answer one question per problem, optionally with the slots replaced."""
    if cfg.kind == "slots":
        toks = np.concatenate([nets._slotted(probs, cfg, True, [[] for _ in probs]),
                               nets._slotted(probs, cfg, False, questions)], axis=1)
    else:
        raise ValueError("slot networks only")
    if slots is None:
        logits, _ = nets.forward(params, jnp.asarray(toks), cfg)
    else:
        logits, _ = nets.forward(params, jnp.asarray(toks), cfg, 0, jnp.arange(K), jnp.asarray(slots), EYE)
    return np.asarray(logits).argmax(-1)


# ------------------------------------------------------------------ the probe
def probe_init(key, d_in):
    k = jax.random.split(key, 3)
    return {"w1": jax.random.normal(k[0], (d_in, 256)) / np.sqrt(d_in), "b1": jnp.zeros(256),
            "w2": jax.random.normal(k[1], (256, 256)) / 16, "b2": jnp.zeros(256),
            "w3": jax.random.normal(k[2], (256, 5)) / 16, "b3": jnp.zeros(5)}


def probe_logits(p, z, letter_onehot):
    h = jax.nn.relu(jnp.concatenate([z, letter_onehot], -1) @ p["w1"] + p["b1"])
    h = jax.nn.relu(h @ p["w2"] + p["b2"])
    return h @ p["w3"] + p["b3"]


def train_probe(Z, probs, shuffle=False, steps=3000, seed=0):
    rng = np.random.default_rng(seed)
    rows, letters, ranks = [], [], []
    for i, p in enumerate(probs):
        order = list(p.order)
        if shuffle:  # control: ranks permuted within each problem, so nothing real can be read
            rng.shuffle(order)
        for r, letter in enumerate(order):
            rows.append(i); letters.append(letter); ranks.append(r)
    rows, letters, ranks = np.array(rows), np.array(letters), np.array(ranks)
    onehot = np.eye(W.LETTERS, dtype=np.float32)
    params = probe_init(jax.random.PRNGKey(seed), Z.shape[1] + W.LETTERS)

    def loss(p, z, l, t):
        lg = probe_logits(p, z, l)
        return -jnp.take_along_axis(jax.nn.log_softmax(lg), t[:, None], 1).mean() + \
            1e-4 * sum(jnp.sum(v ** 2) for k, v in p.items() if k.startswith("w"))

    @jax.jit
    def step(p, m, v, t, z, l, y):
        g = jax.grad(loss)(p, z, l, y)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        p = jax.tree_util.tree_map(lambda a, b, c: a - 1e-3 * (b / (1 - 0.9 ** t)) / (jnp.sqrt(c / (1 - 0.999 ** t)) + 1e-8), p, m, v)
        return p, m, v

    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)
    for t in range(1, steps + 1):
        idx = rng.integers(0, len(rows), 512)
        params, m, v = step(params, m, v, t, jnp.asarray(Z[rows[idx]]), jnp.asarray(onehot[letters[idx]]),
                            jnp.asarray(ranks[idx]))
    return params


def probe_accuracy(params, Z, probs):
    onehot = np.eye(W.LETTERS, dtype=np.float32)
    hits = n = 0
    for letter_pos in range(5):
        letters = np.array([p.order[letter_pos] for p in probs])
        pred = np.asarray(probe_logits(params, jnp.asarray(Z), jnp.asarray(onehot[letters]))).argmax(-1)
        hits += (pred == letter_pos).sum(); n += len(probs)
    return hits / n


# ------------------------------------------------------------------ editing
def edit(params, z0, probs, targets, steps=150, lr=0.05, keep=0.02):
    """Gradient steps on z (normalised slots) until the probe reads the target ranks for all letters."""
    onehot = np.eye(W.LETTERS, dtype=np.float32)
    L = jnp.asarray(np.stack([[onehot[l] for l in p.order] for p in probs]))      # [B, 5, 26]
    T = jnp.asarray(np.stack([[targets[i][l] for l in p.order] for i, p in enumerate(probs)]))  # [B, 5]
    z0 = jnp.asarray(z0)

    def objective(z):
        lg = jax.vmap(lambda zz, ll: probe_logits(params, jnp.broadcast_to(zz, (5, zz.shape[0])), ll))(z, L)
        ce = -jnp.take_along_axis(jax.nn.log_softmax(lg), T[..., None], -1).mean()
        return ce + keep * jnp.mean((z - z0) ** 2)

    grad = jax.jit(jax.grad(objective))
    z, m, v = z0, jnp.zeros_like(z0), jnp.zeros_like(z0)
    for t in range(1, steps + 1):
        g = grad(z)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        z = z - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return np.asarray(z)


def main():
    rng = np.random.default_rng(51)
    train_probs = [W.sample(rng, 5) for _ in range(8000)]
    base, cf = [], []
    while len(base) < 800:
        p = W.sample(rng, 5)
        c = W.rank_counterfactual(p, rng)
        if c is not None:
            base.append(p); cf.append(c)
    # a question about two letters that the swap of x and z does not move, relation unchanged
    other_q, keep_idx = [], []
    for i, (p, c) in enumerate(zip(base, cf)):
        x, y = p.query
        z = next(l for l in c.order if c.rank[l] == p.rank[x])
        rest = [l for l in p.order if l not in (x, z)]
        if len(rest) >= 2:
            u, w = rest[0], rest[1]
            other_q.append((u, w)); keep_idx.append(i)
        else:
            other_q.append((rest[0], y) if rest[0] != y else p.query)
            keep_idx.append(i)
    targets = [c.rank for c in cf]

    for name, path in NETS.items():
        cfg, params, _ = nets.load(ROOT / ".cache/detect" / path)
        S_train = slots_of(params, cfg, train_probs).reshape(len(train_probs), -1)
        mu, sd = S_train.mean(0), S_train.std(0) + 1e-6
        Z_train = (S_train - mu) / sd
        real = train_probe(Z_train, train_probs)
        sham = train_probe(Z_train, train_probs, shuffle=True)
        S_base = slots_of(params, cfg, base).reshape(len(base), -1)
        S_cf = slots_of(params, cfg, cf)
        Z_base = (S_base - mu) / sd
        Z_test = Z_base
        print(f"\n{name}")
        print(f"  probe rank accuracy on held-out slots: real {probe_accuracy(real, Z_test, base):.2f}, "
              f"shuffled-label probe {probe_accuracy(sham, Z_test, base):.2f}")

        lq = [[p.query] for p in base]
        base_ans = answer(params, cfg, base, lq)
        cf_ans = W.labels(cf)
        ok = (base_ans == W.labels(base)) & (answer(params, cfg, cf, [[c.query] for c in cf]) == cf_ans)
        o_q = [[q] for q in other_q]
        o_base = answer(params, cfg, base, o_q)

        def report(label, S_new):
            S_new = S_new.reshape(len(base), K, D)
            flip = (answer(params, cfg, base, lq, S_new) == cf_ans)[ok].mean()
            spec = (answer(params, cfg, base, o_q, S_new) == o_base)[ok].mean()
            if getattr(cfg, "position_questions", 0.0) > 0:  # only networks that know position questions
                pos_q = [[p.query, ("pos", p.query[0], c.rank[p.query[0]])] for p, c in zip(base, cf)]
                pos = f"{(answer(params, cfg, base, pos_q, S_new) == 1)[ok].mean():.2f}"
            else:
                pos = "n/a"
            print(f"  {label:<44} flip {flip:.2f}   other pair unchanged {spec:.2f}   'x at new position' yes {pos}")
            return S_new

        print(f"  evaluated on {ok.sum()} puzzles the network answers correctly in both worlds")
        report("no edit", S_base)
        report("transplant the real counterfactual slots (ceiling)", S_cf)
        Z_real = edit(real, Z_base, base, targets)
        S_real = Z_real * sd + mu
        report("edit guided by the real probe", S_real)
        Z_sham = edit(sham, Z_base, base, targets)
        report("edit guided by the shuffled-label probe", Z_sham * sd + mu)
        delta = np.linalg.norm(S_real - S_base, axis=1, keepdims=True)
        u = np.random.default_rng(3).normal(size=S_base.shape)
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        report("random edit of the same size", S_base + delta * u)
        print(f"  mean edit size relative to the slots: {float((delta[:, 0] / np.linalg.norm(S_base, axis=1)).mean()):.2f}")
        after = probe_accuracy(real, Z_real, cf)
        print(f"  the real probe reads the counterfactual ranks from the edited slots: {after:.2f}")


if __name__ == "__main__":
    main()
