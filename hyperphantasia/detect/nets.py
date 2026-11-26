"""Small transformers whose internals can be read and edited (JAX, CPU).

Two architectures:

* ``plain``  -- a decoder-only transformer over the whole sequence. If it has
  an imagination, the training process had to invent one in its mechanisms.
* ``slots``  -- "Phantasm Slots": an encoder writes the premises into K slot
  vectors, and the question is answered by a decoder that can only see those
  slots. Any scene must pass through the slots, so the architecture supplies
  a place for imagination; the detector checks whether it is actually used.

Both expose every residual-stream vector and accept a patch: replace (part
of) the activation at one layer and some positions, then continue the run.
"""

from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from . import world as W


@dataclass(frozen=True)
class NetConfig:
    kind: str = "plain"      # "plain" | "slots"
    d: int = 64
    heads: int = 4
    layers: int = 4          # plain: total blocks; slots: decoder blocks
    enc_layers: int = 3      # slots only
    k_slots: int = 8         # slots only
    seq_len: int = 1 + W.MAX_PREMISE_TOKENS + 4
    prem_len: int = 1 + W.MAX_PREMISE_TOKENS

    @property
    def vocab(self) -> int:
        return W.BASE_VOCAB + (self.k_slots if self.kind == "slots" else 0)


# ------------------------------------------------------------------- blocks
def _dense(key, fan_in, fan_out):
    return jax.random.normal(key, (fan_in, fan_out)) / math.sqrt(fan_in)


def _block_params(key, d):
    k = jax.random.split(key, 4)
    return {"ln1": (jnp.ones(d), jnp.zeros(d)), "wqkv": _dense(k[0], d, 3 * d), "wo": _dense(k[1], d, d) * 0.5,
            "ln2": (jnp.ones(d), jnp.zeros(d)), "w1": _dense(k[2], d, 4 * d), "b1": jnp.zeros(4 * d),
            "w2": _dense(k[3], 4 * d, d) * 0.5, "b2": jnp.zeros(d)}


def _ln(x, gb):
    g, b = gb
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / jnp.sqrt(var + 1e-5) * g + b


def _block(p, h, bias, heads):
    B, L, d = h.shape
    x = _ln(h, p["ln1"])
    q, k, v = jnp.split(x @ p["wqkv"], 3, axis=-1)
    split = lambda t: t.reshape(B, L, heads, d // heads).transpose(0, 2, 1, 3)
    q, k, v = split(q), split(k), split(v)
    att = jax.nn.softmax(q @ k.transpose(0, 1, 3, 2) / math.sqrt(d // heads) + bias[:, None], axis=-1)
    o = (att @ v).transpose(0, 2, 1, 3).reshape(B, L, d)
    h = h + o @ p["wo"]
    x = _ln(h, p["ln2"])
    return h + jax.nn.gelu(x @ p["w1"] + p["b1"]) @ p["w2"] + p["b2"]


def _bias(key_ok):
    """Causal mask that also hides padding keys."""
    L = key_ok.shape[1]
    causal = jnp.tril(jnp.ones((L, L), dtype=bool))
    ok = causal[None] & key_ok[:, None, :]
    return jnp.where(ok, 0.0, -1e9)


def _patch(h, pos, vals, proj):
    cur = h[:, pos]
    return h.at[:, pos].set(cur + (vals - cur) @ proj)


# ---------------------------------------------------------------- networks
def init(cfg: NetConfig, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 8)
    p = {"tok": jax.random.normal(ks[0], (cfg.vocab, cfg.d)) * 0.5,
         "lnf": (jnp.ones(cfg.d), jnp.zeros(cfg.d)), "head": _dense(ks[1], cfg.d, 2)}
    if cfg.kind == "plain":
        p["pos"] = jax.random.normal(ks[2], (cfg.seq_len, cfg.d)) * 0.5
        p["blocks"] = [_block_params(k, cfg.d) for k in jax.random.split(ks[3], cfg.layers)]
    else:
        p["pos_enc"] = jax.random.normal(ks[2], (cfg.prem_len + cfg.k_slots, cfg.d)) * 0.5
        p["pos_dec"] = jax.random.normal(ks[4], (cfg.k_slots + 4, cfg.d)) * 0.5
        p["enc"] = [_block_params(k, cfg.d) for k in jax.random.split(ks[3], cfg.enc_layers)]
        p["blocks"] = [_block_params(k, cfg.d) for k in jax.random.split(ks[5], cfg.layers)]
    return p


@partial(jax.jit, static_argnames=("cfg", "patch_layer"))
def forward(params, tokens, cfg: NetConfig, patch_layer: int = -1, patch_pos=None, patch_vals=None, proj=None):
    """Returns (logits [B,2], residuals: list over layers of [B, L, d]).

    For ``plain`` the residual positions are the sequence positions. For
    ``slots`` they are the decoder positions: K slot vectors, then ? x y =.
    """
    if cfg.kind == "plain":
        h = params["tok"][tokens] + params["pos"][None]
        bias = _bias(tokens != W.PAD)
    else:
        prem, query = tokens[:, :cfg.prem_len], tokens[:, cfg.prem_len:]
        slot_ids = jnp.broadcast_to(jnp.arange(W.BASE_VOCAB, cfg.vocab), (tokens.shape[0], cfg.k_slots))
        enc_tok = jnp.concatenate([prem, slot_ids], axis=1)
        e = params["tok"][enc_tok] + params["pos_enc"][None]
        ebias = _bias(enc_tok != W.PAD)
        for bp in params["enc"]:
            e = _block(bp, e, ebias, cfg.heads)
        slots = e[:, cfg.prem_len:]
        h = jnp.concatenate([slots, params["tok"][query]], axis=1) + params["pos_dec"][None]
        bias = _bias(jnp.ones(h.shape[:2], dtype=bool))
    if patch_layer == 0:
        h = _patch(h, patch_pos, patch_vals, proj)
    resid = [h]
    for i, bp in enumerate(params["blocks"]):
        h = _block(bp, h, bias, cfg.heads)
        if patch_layer == i + 1:
            h = _patch(h, patch_pos, patch_vals, proj)
        resid.append(h)
    logits = _ln(h[:, -1], params["lnf"]) @ params["head"]
    return logits, resid


def encode(cfg: NetConfig, problems) -> np.ndarray:
    if cfg.kind == "plain":
        return W.encode_sequence(problems, cfg.seq_len)
    q = np.array([[W.QMARK, p.query[0], p.query[1], W.ANS] for p in problems], dtype=np.int32)
    return np.concatenate([W.encode_premises(problems, cfg.prem_len), q], axis=1)


def roles(cfg: NetConfig) -> dict[str, list[int]]:
    """Residual positions that play each role in the question block."""
    if cfg.kind == "plain":
        L = cfg.seq_len
        return {"scene": [L - 4], "X": [L - 3], "Y": [L - 2], "ANS": [L - 1]}
    K = cfg.k_slots
    return {"scene": list(range(K)), "X": [K + 1], "Y": [K + 2], "ANS": [K + 3]}


# ----------------------------------------------------------------- training
@dataclass
class TrainLog:
    steps: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    acc: list[float] = field(default_factory=list)


def _loss(params, tokens, y, cfg):
    logits, _ = forward(params, tokens, cfg)
    logp = jax.nn.log_softmax(logits)
    return -jnp.take_along_axis(logp, y[:, None], axis=1).mean()


@partial(jax.jit, static_argnames=("cfg",))
def _step(params, opt, tokens, y, lr, cfg):
    loss, g = jax.value_and_grad(_loss)(params, tokens, y, cfg)
    t, m, v = opt
    t = t + 1
    m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
    v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
    mhat = 1 - 0.9 ** t
    vhat = 1 - 0.999 ** t
    params = jax.tree_util.tree_map(lambda p, a, b: p - lr * (a / mhat) / (jnp.sqrt(b / vhat) + 1e-8),
                                    params, m, v)
    return params, (t, m, v), loss


def accuracy(params, cfg, problems) -> float:
    logits, _ = forward(params, jnp.asarray(encode(cfg, problems)), cfg)
    return float((np.asarray(logits).argmax(-1) == W.labels(problems)).mean())


def train(cfg: NetConfig, steps: int = 6000, batch: int = 128, lr: float = 2e-3, seed: int = 0,
          log_every: int = 500, target: float = 0.995, verbose: bool = True,
          checkpoint: Path | None = None):
    """Train on answers only. With ``checkpoint``, progress is saved and resumed."""
    rng = np.random.default_rng(seed)
    params = init(cfg, seed)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt = (0, zeros, zeros)
    val = W.batch(np.random.default_rng(seed + 1000), 2000)
    log = TrainLog()
    first = 1
    if checkpoint and Path(checkpoint).exists():
        with open(checkpoint, "rb") as f:
            ck = pickle.load(f)
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        opt = (ck["opt"][0], *(jax.tree_util.tree_map(jnp.asarray, x) for x in ck["opt"][1:]))
        rng, log, first = ck["rng"], ck["log"], ck["step"] + 1
        if verbose:
            print(f"  [{cfg.kind}] resumed at step {first}", flush=True)
    for s in range(first, steps + 1):
        # Curriculum on world size only (2 letters first, up to 6). Nothing about the
        # internal scene is supervised: only the yes/no answer is ever trained.
        max_n = min(6, 3 + int(4 * s / (0.6 * steps)))
        probs = W.batch(rng, batch, ns=tuple(range(2, max_n + 1)))
        cur_lr = lr * min(1.0, s / 200) * (0.5 * (1 + math.cos(math.pi * s / steps)) * 0.9 + 0.1)
        params, opt, loss = _step(params, opt, jnp.asarray(encode(cfg, probs)),
                                  jnp.asarray(W.labels(probs)), cur_lr, cfg)
        if s % log_every == 0 or s == steps:
            acc = accuracy(params, cfg, val)
            log.steps.append(s); log.loss.append(float(loss)); log.acc.append(acc)
            if verbose:
                print(f"  [{cfg.kind}] step {s:>5}  loss {float(loss):.4f}  val acc {acc:.3f}", flush=True)
            if checkpoint:
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint, "wb") as f:
                    pickle.dump({"params": jax.tree_util.tree_map(np.asarray, params), "step": s,
                                 "opt": (int(opt[0]), *(jax.tree_util.tree_map(np.asarray, x) for x in opt[1:])),
                                 "rng": rng, "log": log}, f)
            if acc >= target and s >= 1000:
                break
    return params, log


def save(path: Path, cfg: NetConfig, params, log: TrainLog) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"cfg": cfg, "params": jax.tree_util.tree_map(np.asarray, params), "log": log}, f)


def load(path: Path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["cfg"], jax.tree_util.tree_map(jnp.asarray, blob["params"]), blob["log"]
