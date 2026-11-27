"""Is the readable scene outside the learned causal subspace idle?

Retrains the k=8 subspace of das_slots.py (same seeds), then measures interchange accuracy when
transplanting only the learned subspace, only its complement (the other 376 dimensions, which hold
the readable scene), and both, on held-out pairs whose answers differ. Saves the subspace.

    python experiments/detector_followups/das_complement.py
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import das_slots as d  # noqa: E402

if __name__ == "__main__":
    U = d.train_subspace(8)
    np.save(d.ROOT / ".cache/detect/das_slots_k8.npy", np.asarray(U))
    Ucomp, _ = jnp.linalg.qr(jnp.eye(d.DIM) - U @ U.T)  # spans the complement (rank 376) plus null dirs
    P = jnp.eye(d.DIM) - U @ U.T
    print("answer follows the source world, on held-out pairs whose answers differ:")
    print(f"  transplant the learned 8 dims only           {d.iia(U):.3f}")
    rng = np.random.default_rng(99)
    base, src, qs = d.make_batch(rng, 2000)
    toks = d.tokens(base, [q for q, _, _ in qs], rng)
    hb, hs = d.slots(base), d.slots(src)
    tgt = np.array([t for _, t, _ in qs]); differs = np.array([t != b for _, t, b in qs])
    for label, M in (("transplant the other 376 dims only", P), ("transplant everything", jnp.eye(d.DIM))):
        patched = hb + (hs - hb) @ M
        logits, _ = d.nets.forward(d.params, toks, d.cfg, 0, jnp.arange(d.K), patched.reshape(-1, d.K, d.D), d.EYE)
        print(f"  {label:<44} {float((np.asarray(logits).argmax(-1) == tgt)[differs].mean()):.3f}")
