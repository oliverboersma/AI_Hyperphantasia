"""Exploratory: is x's position computed only when the question asks for it?

For the two networks trained on mixed questions, put a second question after the usual left-of
question, either "is x left of y?" or "is x at position k?" (same x), and probe x's rank from x's
token in that second question at every layer. Ridge probes as in the detector; chance 0.20, surface
counts 0.59.

    python experiments/detector_followups/question_context.py
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, probes, world as W  # noqa: E402

NETS = {"plain, mixed questions": "plain-pos_seed0_steps8000_curriculum.pkl",
        "bottleneck, mixed questions (door shut)": "side-door-pos-shut-continued_seed0_steps6000.pkl"}


def tokens(cfg, probs, second):
    if cfg.kind == "plain":
        return nets._slotted(probs, cfg, True, [[p.query, q] for p, q in zip(probs, second)])
    return np.concatenate([nets._slotted(probs, cfg, True, [[] for _ in probs]),
                           nets._slotted(probs, cfg, False, [[p.query, q] for p, q in zip(probs, second)])], axis=1)


if __name__ == "__main__":
    rng = np.random.default_rng(31)
    probs = [W.sample(rng, 5) for _ in range(6000)]
    xs = [int(rng.choice(p.order)) for p in probs]
    ys = [int(rng.choice([z for z in p.order if z != x])) for p, x in zip(probs, xs)]
    ks = [int(rng.integers(0, 5)) for _ in probs]
    target = np.array([p.rank[x] for p, x in zip(probs, xs)])
    for name, path in NETS.items():
        cfg, params, _ = nets.load(ROOT / ".cache/detect" / path)
        print(name)
        for label, second in (("second question 'is x left of y?'", [(x, y) for x, y in zip(xs, ys)]),
                              ("second question 'is x at position k?'", [("pos", x, k) for x, k in zip(xs, ks)])):
            toks = tokens(cfg, probs, second)
            _, resid = nets.forward(params, jnp.asarray(toks), cfg)
            x_pos = len(resid[0][0]) - 3  # the x token of the last question block
            accs = [probes.fit(np.asarray(r)[:, x_pos], target, 5, seed=0).acc for r in resid]
            print(f"   {label:<40} rank of x readable at x's token, layers 0-4: "
                  + "  ".join(f"{a:.2f}" for a in accs))
