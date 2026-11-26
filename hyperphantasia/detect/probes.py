"""Linear probes (ridge regression onto one-hot labels) and their subspaces."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Probe:
    W: np.ndarray       # [d, C]
    mu: np.ndarray      # [d]
    classes: int
    acc: float          # held-out accuracy

    def predict(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mu) @ self.W).argmax(-1)

    def subspace(self, tol: float = 1e-6) -> np.ndarray:
        """Orthonormal basis [d, k] of the directions the probe reads."""
        Wc = self.W - self.W.mean(axis=1, keepdims=True)
        u, s, _ = np.linalg.svd(Wc, full_matrices=False)
        return u[:, s > tol * max(s.max(), 1e-12)]


def _ridge(X, Y, lam):
    d = X.shape[1]
    if X.shape[0] >= d:
        return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)
    return X.T @ np.linalg.solve(X @ X.T + lam * np.eye(X.shape[0]), Y)  # dual form for wide inputs


def fit(X: np.ndarray, y: np.ndarray, classes: int, seed: int = 0) -> Probe:
    """Fit on 60%, choose the ridge strength on 20%, report accuracy on the last 20%."""
    X = X.reshape(len(X), -1).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    a, b = int(0.6 * len(X)), int(0.8 * len(X))
    tr, va, te = idx[:a], idx[a:b], idx[b:]
    mu = X[tr].mean(0)
    Y = np.eye(classes)[y]
    scale = max(float((X[tr] - mu).var(0).sum()), 1e-9)
    best = None
    for lam in (1e-4, 1e-3, 1e-2, 1e-1, 1.0):
        Wt = _ridge(X[tr] - mu, Y[tr] - Y[tr].mean(0), lam * scale * len(tr) / X.shape[1])
        acc = float((((X[va] - mu) @ Wt).argmax(-1) == y[va]).mean())
        if best is None or acc > best[0]:
            best = (acc, Wt)
    Wt = best[1]
    test = float((((X[te] - mu) @ Wt).argmax(-1) == y[te]).mean())
    return Probe(Wt, mu, classes, test)


def random_subspace(d: int, k: int, rng) -> np.ndarray:
    q, _ = np.linalg.qr(rng.normal(size=(d, k)))
    return q
