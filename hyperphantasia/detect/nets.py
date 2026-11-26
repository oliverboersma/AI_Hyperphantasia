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
    # "ids": each premise's three tokens share a premise id drawn at random per puzzle, plus a
    # role tag (left letter, sign, right letter). Grouping is free and no position is special,
    # so position-bound shortcuts are impossible. "absolute": learned absolute positions.
    encoding: str = "ids"
    # Training only ever supervises answers. Each training sequence asks ``train_queries``
    # questions about the same premises (more answer bits per pass); evaluation asks one,
    # and causal attention means later questions cannot affect the first answer.
    train_queries: int = 4
    max_train_n: int = 5

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
N_ROLES, N_PIDS = 10, 9  # roles: pad, BOS, left, sign, right, ?, x, y, =, slot; ids: none + 8


def init(cfg: NetConfig, seed: int = 0):
    key = jax.random.PRNGKey(seed)
    ks = jax.random.split(key, 10)
    p = {"tok": jax.random.normal(ks[0], (cfg.vocab, cfg.d)) * 0.5,
         "lnf": (jnp.ones(cfg.d), jnp.zeros(cfg.d)), "head": _dense(ks[1], cfg.d, 2)}
    ids = getattr(cfg, "encoding", "absolute") == "ids"
    if ids:
        p["role"] = jax.random.normal(ks[6], (N_ROLES, cfg.d)) * 0.5
        p["pid"] = jax.random.normal(ks[7], (N_PIDS, cfg.d)) * 0.5
    if cfg.kind == "plain":
        if not ids:
            p["pos"] = jax.random.normal(ks[2], (cfg.seq_len, cfg.d)) * 0.5
        p["blocks"] = [_block_params(k, cfg.d) for k in jax.random.split(ks[3], cfg.layers)]
    else:
        if not ids:
            p["pos_enc"] = jax.random.normal(ks[2], (cfg.prem_len + cfg.k_slots, cfg.d)) * 0.5
        p["pos_dec"] = jax.random.normal(ks[4], (cfg.k_slots + (0 if ids else 4), cfg.d)) * 0.5
        p["enc"] = [_block_params(k, cfg.d) for k in jax.random.split(ks[3], cfg.enc_layers)]
        p["blocks"] = [_block_params(k, cfg.d) for k in jax.random.split(ks[5], cfg.layers)]
    return p


def _trunk(params, tokens, cfg: NetConfig, patch_layer: int = -1, patch_pos=None, patch_vals=None, proj=None):
    ids = getattr(cfg, "encoding", "absolute") == "ids"
    embed = (lambda t: params["tok"][t[..., 0]] + params["role"][t[..., 1]] + params["pid"][t[..., 2]]) \
        if ids else (lambda t: params["tok"][t])
    word = (lambda t: t[..., 0]) if ids else (lambda t: t)
    if cfg.kind == "plain":
        h = embed(tokens) + (0.0 if ids else params["pos"][None])
        bias = _bias(word(tokens) != W.PAD)
    else:
        prem, query = tokens[:, :cfg.prem_len], tokens[:, cfg.prem_len:]
        slot_ids = jnp.broadcast_to(jnp.arange(W.BASE_VOCAB, cfg.vocab), (tokens.shape[0], cfg.k_slots))
        slot_emb = params["tok"][slot_ids] + (params["role"][N_ROLES - 1] if ids else 0.0)
        e = jnp.concatenate([embed(prem), slot_emb], axis=1) + (0.0 if ids else params["pos_enc"][None])
        ebias = _bias(jnp.concatenate([word(prem) != W.PAD, jnp.ones_like(slot_ids, dtype=bool)], axis=1))
        for bp in params["enc"]:
            e = _block(bp, e, ebias, cfg.heads)
        slots = e[:, cfg.prem_len:]
        if ids:  # slots carry their own position; question tokens carry role tags
            h = jnp.concatenate([slots + params["pos_dec"][None], embed(query)], axis=1)
        else:
            h = jnp.concatenate([slots, embed(query)], axis=1) + params["pos_dec"][None]
        bias = _bias(jnp.ones(h.shape[:2], dtype=bool))
    if patch_layer == 0:
        h = _patch(h, patch_pos, patch_vals, proj)
    resid = [h]
    for i, bp in enumerate(params["blocks"]):
        h = _block(bp, h, bias, cfg.heads)
        if patch_layer == i + 1:
            h = _patch(h, patch_pos, patch_vals, proj)
        resid.append(h)
    return h, resid


@partial(jax.jit, static_argnames=("cfg", "patch_layer"))
def forward(params, tokens, cfg: NetConfig, patch_layer: int = -1, patch_pos=None, patch_vals=None, proj=None):
    """Returns (logits [B,2] at the last position, residuals: list over layers of [B, L, d]).

    For ``plain`` the residual positions are the sequence positions. For
    ``slots`` they are the decoder positions: K slot vectors, then ? x y =.
    """
    h, resid = _trunk(params, tokens, cfg, patch_layer, patch_pos, patch_vals, proj)
    return _ln(h[:, -1], params["lnf"]) @ params["head"], resid


def _ids(problems, length: int, premises: bool, query: bool, extra=None) -> np.ndarray:
    out = np.zeros((len(problems), length, 3), dtype=np.int32)
    out[..., 0] = W.PAD
    for i, p in enumerate(problems):
        rows = []
        if premises:
            rng = np.random.default_rng(abs(hash(p.premises)) % 2 ** 32)  # deterministic per puzzle
            pids = rng.permutation(np.arange(1, N_PIDS))
            rows.append((W.BOS, 1, 0))
            for k, (a, op, b) in enumerate(p.premises):
                rows += [(a, 2, pids[k]), (op, 3, pids[k]), (b, 4, pids[k])]
        if query:
            for x, y in [p.query] + list(extra[i] if extra else []):
                rows += [(W.QMARK, 5, 0), (x, 6, 0), (y, 7, 0), (W.ANS, 8, 0)]
        out[i, length - len(rows):] = rows
    return out


def encode_train(cfg: NetConfig, problems, rng):
    """Training sequences with several questions; returns (tokens, labels, mask) per position."""
    q = cfg.train_queries
    extra = []
    for p in problems:
        pairs = [tuple(int(v) for v in rng.choice(p.order, size=2, replace=False)) for _ in range(q - 1)]
        extra.append(pairs)
    if cfg.kind == "plain":  # no positions are used, so the shortest padding will do
        toks = _ids(problems, 1 + 3 * (cfg.max_train_n - 1) + 4 * q, True, True, extra)
        offset = 0
    else:
        prem_len = cfg.prem_len  # the network splits premises from questions here
        toks = np.concatenate([_ids(problems, prem_len, True, False),
                               _ids(problems, 4 * q, False, True, extra)], axis=1)
        offset = prem_len - cfg.k_slots  # decoder positions = K slots, then the questions
    L = toks.shape[1] - offset
    y = np.zeros((len(problems), L), dtype=np.int32)
    mask = np.zeros((len(problems), L), dtype=np.float32)
    for i, p in enumerate(problems):
        r = p.rank
        for j, (a, b) in enumerate([p.query] + extra[i]):
            pos = L - 1 - 4 * (q - 1 - j)
            y[i, pos], mask[i, pos] = int(r[a] < r[b]), 1.0
    return toks, y, mask


def encode(cfg: NetConfig, problems) -> np.ndarray:
    if getattr(cfg, "encoding", "absolute") == "ids":
        if cfg.kind == "plain":
            return _ids(problems, cfg.seq_len, True, True)
        return np.concatenate([_ids(problems, cfg.prem_len, True, False), _ids(problems, 4, False, True)], axis=1)
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


def _loss(params, tokens, y, mask, cfg):
    h, _ = _trunk(params, tokens, cfg)
    logp = jax.nn.log_softmax(_ln(h, params["lnf"]) @ params["head"])  # [B, L, 2]
    nll = -jnp.take_along_axis(logp, y[..., None], axis=-1)[..., 0]
    return (nll * mask).sum() / mask.sum()


@partial(jax.jit, static_argnames=("cfg",))
def _step(params, opt, tokens, y, mask, lr, cfg):
    loss, g = jax.value_and_grad(_loss)(params, tokens, y, mask, cfg)
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


def train(cfg: NetConfig, steps: int = 8000, batch: int = 256, lr: float = 2e-3, seed: int = 0,
          log_every: int = 500, target: float = 0.995, verbose: bool = True,
          checkpoint: Path | None = None, curriculum: bool = False):
    """Train on answers only. With ``checkpoint``, progress is saved and resumed.

    By default every batch mixes worlds of 2 to 6 letters. ``curriculum=True``
    starts with tiny worlds and grows them; in our runs that taught the plain
    transformer a position-bound shortcut that breaks on larger worlds.
    """
    rng = np.random.default_rng(seed)
    params = init(cfg, seed)
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    opt = (0, zeros, zeros)
    val = W.batch(np.random.default_rng(seed + 1000), 2000, ns=tuple(range(3, cfg.max_train_n + 1)))
    log = TrainLog()
    first = 1
    if checkpoint and Path(checkpoint).exists():
        with open(checkpoint, "rb") as f:
            ck = pickle.load(f)
        if ck.get("cfg") != cfg or ck.get("curriculum") != curriculum:
            raise ValueError(f"{checkpoint} was saved for a different network; delete it to start over")
        params = jax.tree_util.tree_map(jnp.asarray, ck["params"])
        opt = (ck["opt"][0], *(jax.tree_util.tree_map(jnp.asarray, x) for x in ck["opt"][1:]))
        rng, log, first = ck["rng"], ck["log"], ck["step"] + 1
        if verbose:
            print(f"  [{cfg.kind}] resumed at step {first}", flush=True)
    for s in range(first, steps + 1):
        # Only the yes/no answer is ever trained; nothing about the scene is supervised.
        top = cfg.max_train_n
        max_n = min(top, 3 + int((top - 2) * s / (0.6 * steps))) if curriculum else top
        probs = W.batch(rng, batch, ns=tuple(range(2, max_n + 1)))
        cur_lr = lr * min(1.0, s / 200) * (0.5 * (1 + math.cos(math.pi * s / steps)) * 0.9 + 0.1)
        toks, y, mask = encode_train(cfg, probs, rng) if cfg.encoding == "ids" else \
            (encode(cfg, probs), None, None)
        if y is None:  # older absolute-position networks: one question per sequence
            y = np.zeros(toks.shape[:2], dtype=np.int32); y[:, -1] = W.labels(probs)
            mask = np.zeros(toks.shape[:2], dtype=np.float32); mask[:, -1] = 1.0
            if cfg.kind == "slots":
                y, mask = y[:, cfg.prem_len - cfg.k_slots:], mask[:, cfg.prem_len - cfg.k_slots:]
        params, opt, loss = _step(params, opt, jnp.asarray(toks), jnp.asarray(y), jnp.asarray(mask),
                                  cur_lr, cfg)
        if s % log_every == 0 or s == steps:
            acc = accuracy(params, cfg, val)
            log.steps.append(s); log.loss.append(float(loss)); log.acc.append(acc)
            if verbose:
                print(f"  [{cfg.kind}] step {s:>5}  loss {float(loss):.4f}  val acc {acc:.3f}", flush=True)
            if checkpoint:
                Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint, "wb") as f:
                    pickle.dump({"cfg": cfg, "curriculum": curriculum,
                                 "params": jax.tree_util.tree_map(np.asarray, params), "step": s,
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
    cfg = blob["cfg"]
    if "encoding" not in cfg.__dict__:  # saved before these options existed
        object.__setattr__(cfg, "encoding", "absolute")
        object.__setattr__(cfg, "train_queries", 1)
        object.__setattr__(cfg, "max_train_n", 6)
    return cfg, jax.tree_util.tree_map(jnp.asarray, blob["params"]), blob["log"]
