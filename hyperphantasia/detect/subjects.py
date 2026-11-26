"""Subjects: anything the detector can examine.

A subject answers problems and exposes its internal activations at named
sites ``(layer, role)``, where role is one of

    scene  -- what exists after the premises, before the question is read
    X, Y   -- the positions of the two asked-about letters
    ANS    -- where the answer is produced

and it lets the detector overwrite (part of) the activation at a site and
continue the computation. Two subjects are built by hand so the detector can
be calibrated against known ground truth:

* ``SceneEngine``     has a real internal scene (each letter's rank) and reads
                      its answers off it. The detector must say: imagination.
* ``SurfaceHeuristic`` never composes premises: it looks up stated pairs and
                      otherwise compares surface counts. The detector must
                      say: no imagination.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import world as W

ROLES = ("scene", "X", "Y", "ANS")


@dataclass
class Patch:
    layer: int
    role: str
    values: np.ndarray            # [B, P, d] activations to transplant
    proj: np.ndarray | None = None  # [d, d] projector: patch only this subspace (None = everything)


class Subject:
    name = "subject"
    layers: list[int] = [1]

    def run(self, problems, patch: Patch | None = None):
        """Return (predicted answers [B], activations {(layer, role): [B, P, d]})."""
        raise NotImplementedError


def _onehot(i, n):
    v = np.zeros(n, dtype=np.float32)
    v[i] = 1.0
    return v


def _apply(vec, patch, b):
    proj = patch.proj if patch.proj is not None else np.eye(vec.shape[-1], dtype=np.float32)
    return vec + (patch.values[b, 0] - vec) @ proj


class SceneEngine(Subject):
    """Known positive: builds the scene first, then answers from it."""

    name = "known positive: scene engine"
    R = W.MAX_N

    def run(self, problems, patch=None):
        preds, acts = [], {(1, r): [] for r in ROLES}
        for b, p in enumerate(problems):
            scene = np.zeros(W.LETTERS * self.R, dtype=np.float32)
            for letter, rank in p.rank.items():
                scene[letter * self.R + rank] = 1.0
            if patch and patch.role == "scene":
                scene = _apply(scene, patch, b)
            read = lambda t: int(np.argmax(scene[t * self.R:(t + 1) * self.R]))
            x = _onehot(read(p.query[0]), self.R)
            if patch and patch.role == "X":
                x = _apply(x, patch, b)
            y = _onehot(read(p.query[1]), self.R)
            if patch and patch.role == "Y":
                y = _apply(y, patch, b)
            ans = np.array([float(np.argmax(x) < np.argmax(y))], dtype=np.float32)
            if patch and patch.role == "ANS":
                ans = _apply(ans, patch, b)
            preds.append(int(ans[0] > 0.5))
            for role, v in zip(ROLES, (scene, x, y, ans)):
                acts[(1, role)].append(v[None])
        return np.array(preds), {k: np.stack(v) for k, v in acts.items()}


class SurfaceHeuristic(Subject):
    """Known negative: stated pairs plus surface counts, never composition."""

    name = "known negative: surface heuristic"

    def run(self, problems, patch=None):
        preds, acts = [], {(1, r): [] for r in ROLES}
        for b, p in enumerate(problems):
            feats = W.surface_features(p)
            table = np.zeros((W.LETTERS, 4), dtype=np.float32)
            for letter, f in feats.items():
                table[letter] = f
            adj = np.zeros((W.LETTERS, W.LETTERS), dtype=np.float32)
            for a, op, c in p.premises:
                lo, hi = (a, c) if op == W.LT else (c, a)
                adj[lo, hi] = 1.0
            scene = np.concatenate([table.ravel(), adj.ravel()])
            if patch and patch.role == "scene":
                scene = _apply(scene, patch, b)
            table = scene[:W.LETTERS * 4].reshape(W.LETTERS, 4)
            adj = scene[W.LETTERS * 4:].reshape(W.LETTERS, W.LETTERS)
            qx, qy = p.query
            x = table[qx].copy()
            if patch and patch.role == "X":
                x = _apply(x, patch, b)
            y = table[qy].copy()
            if patch and patch.role == "Y":
                y = _apply(y, patch, b)
            if adj[qx, qy] > 0.5:
                ans = 1.0
            elif adj[qy, qx] > 0.5:
                ans = 0.0
            elif x[3] != y[3]:
                ans = float(x[3] > y[3])          # more "smaller-than" appearances => further left
            else:
                ans = float(x[2] < y[2])          # tie: whichever was mentioned first
            a = np.array([ans], dtype=np.float32)
            if patch and patch.role == "ANS":
                a = _apply(a, patch, b)
            preds.append(int(a[0] > 0.5))
            for role, v in zip(ROLES, (scene, x, y, a)):
                acts[(1, role)].append(v[None])
        return np.array(preds), {k: np.stack(v) for k, v in acts.items()}


class NeuralSubject(Subject):
    """A trained (or untrained, as a control) network from ``nets``."""

    def __init__(self, cfg, params, name: str):
        from . import nets
        self.nets, self.cfg, self.params, self.name = nets, cfg, params, name
        self.layers = list(range(cfg.layers + 1))
        self.positions = nets.roles(cfg)

    def run(self, problems, patch=None, chunk: int = 1000):
        import jax.numpy as jnp
        preds, acts = [], {}
        for start in range(0, len(problems), chunk):
            part = problems[start:start + chunk]
            tokens = jnp.asarray(self.nets.encode(self.cfg, part))
            if patch is None:
                logits, resid = self.nets.forward(self.params, tokens, self.cfg)
            else:
                pos = jnp.asarray(self.positions[patch.role])
                d = self.cfg.d
                proj = jnp.eye(d) if patch.proj is None else jnp.asarray(patch.proj)
                vals = jnp.asarray(patch.values[start:start + chunk])
                logits, resid = self.nets.forward(self.params, tokens, self.cfg, patch.layer, pos, vals, proj)
            preds.append(np.asarray(logits).argmax(-1))
            for layer, h in enumerate(resid):
                h = np.asarray(h)
                for role, pos in self.positions.items():
                    acts.setdefault((layer, role), []).append(h[:, pos])
        return np.concatenate(preds), {k: np.concatenate(v) for k, v in acts.items()}
