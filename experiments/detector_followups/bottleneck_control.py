"""Control for the Phantasm Slots failures: the same encoder-decoder without a bottleneck.

Every slot-bottleneck layout (free slots, anchored slots, Slot Attention, Slot Attention with raw
letter embeddings) stayed at chance. This control keeps the encoder (bidirectional, over the
slotted premise grid) and the 4-block decoder, but lets the decoder attend to *every* encoder
position (padding masked) instead of a few slot vectors. Same early-curriculum worlds (2-3
letters), batch 256, learning rate 2e-3. If this learns, the encoder-decoder machinery works and
the bottleneck itself is what blocks learning.

    python experiments/detector_followups/bottleneck_control.py
Output from our run: bottleneck_control.out.txt
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperphantasia.detect import nets, world as W  # noqa: E402
from hyperphantasia.detect.nets import _bias, _bias_full, _block, _ln  # noqa: E402

cfg = nets.NetConfig(kind="slots", k_slots=nets.NetConfig().prem_len)  # one "slot" per premise position
params = nets.init(cfg, 0)


def trunk(params, tokens):
    prem, query = tokens[:, :cfg.prem_len], tokens[:, cfg.prem_len:]
    e = params["tok"][prem] + params["pos_enc"][None, :cfg.prem_len]
    keys = prem != W.PAD
    eb = _bias_full(keys)
    for bp in params["enc"]:
        e = _block(bp, e, eb, cfg.heads)
    h = jnp.concatenate([e, params["tok"][query]], axis=1)
    h = h + params["pos_dec"][None, :h.shape[1]]
    b = _bias(jnp.concatenate([keys, jnp.ones(query.shape, bool)], axis=1))
    for bp in params["blocks"]:
        h = _block(bp, h, b, cfg.heads)
    return h


def loss(params, toks, y, mask):
    logp = jax.nn.log_softmax(_ln(trunk(params, toks), params["lnf"]) @ params["head"])
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    return (nll * mask).sum() / mask.sum()


@jax.jit
def step(params, opt, toks, y, mask, lr):
    l, g = jax.value_and_grad(loss)(params, toks, y, mask)
    t, m, v = opt
    t = t + 1
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
    v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
    params = jax.tree_util.tree_map(
        lambda p, a, b: p - lr * (a / (1 - 0.9 ** t)) / (jnp.sqrt(b / (1 - 0.999 ** t)) + 1e-8), params, m, v)
    return params, (t, m, v), l


if __name__ == "__main__":
    z = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt = (0, z, z)
    rng = np.random.default_rng(0)
    for s in range(1, 2001):
        probs = W.batch(rng, 256, ns=(2, 3))  # the same early-curriculum worlds
        toks, y, mask = nets.encode_train(cfg, probs, rng)
        params, opt, l = step(params, opt, jnp.asarray(toks), jnp.asarray(y), jnp.asarray(mask),
                              2e-3 * min(1, s / 200))
        if s % 250 == 0:
            print(f"   encoder-decoder, no bottleneck, 2-3 letter worlds: step {s} loss {float(l):.3f}", flush=True)
