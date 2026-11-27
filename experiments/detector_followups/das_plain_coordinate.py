"""Is the plain transformer's one-direction code at x's site a coordinate for x?

Follow-up to das_plain.py (same network, site and pairs).

1. Control for part B. On the reorder pairs where the two hypotheses disagree, how often does the
   base answer itself agree with "coordinate"? If the patch did nothing, "follows coordinate" would
   be that rate, not the 85% measured.
2. The learned one-dimensional subspace (k=1, trained on move-x pairs): mean value by x's rank, and
   rank correlation.
3. Placing x by setting one number. For each puzzle and each rank r other than y's, set x's
   coordinate to its mean value for rank r and leave everything else as it is. A coordinate should
   make the network answer "r < rank(y)", as if x had been put at r. Control: the same with a
   random direction.

    python experiments/detector_followups/das_plain_coordinate.py
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import das_plain as d  # noqa: E402
from hyperphantasia.detect import nets, world as W  # noqa: E402

if __name__ == "__main__":
    base, src, tg = d.EVAL["reorder"]
    rel, coord, ok = tg["relations"][0], tg["coordinate"][0], tg["coordinate"][1]
    keep = ok & (rel != coord)
    none = d.predict("reorder", jnp.zeros((d.D, 1)))
    print(f"1. Reorder pairs where the hypotheses disagree ({keep.sum()}):")
    print(f"   base answer agrees with 'coordinate'      {(tg['base'][0] == coord)[keep].mean():.3f}")
    print(f"   unpatched network agrees with 'coordinate' {(none == coord)[keep].mean():.3f}")
    print(f"   whole-site transplant agrees              {(d.predict('reorder', d.EYE) == coord)[keep].mean():.3f}")

    u = np.asarray(d.train(1, "move_x", "relations"))[:, 0]
    rng = np.random.default_rng(41)
    probs = [W.sample(rng, 5) for _ in range(6000)]
    toks, H = d.site(probs)
    H = np.asarray(H)
    xr = np.array([p.rank[p.query[0]] for p in probs])
    proj = H @ u
    means = np.array([proj[xr == r].mean() for r in range(5)])
    sds = np.array([proj[xr == r].std() for r in range(5)])
    rho = np.corrcoef(np.argsort(np.argsort(proj)), xr)[0, 1]
    print("\n2. The learned direction, by x's rank (5-letter worlds):")
    print("   rank   " + "  ".join(f"{r:>6}" for r in range(5)))
    print("   mean   " + "  ".join(f"{m:6.2f}" for m in means))
    print("   sd     " + "  ".join(f"{s:6.2f}" for s in sds))
    print(f"   rank correlation with x's rank: {rho:.3f}")

    def place(direction, levels):
        """Set x's coordinate along `direction` to levels[r]; score answer == (r < rank(y))."""
        yr = np.array([p.rank[p.query[1]] for p in probs])
        hit = np.zeros((5,)); cnt = np.zeros((5,))
        for r in range(5):
            idx = np.where((yr != r) & (xr != r))[0]  # x actually moves, and r is not y's spot
            hb = H[idx]
            patched = hb + np.outer(levels[r] - hb @ direction, direction)
            logits, _ = nets.forward(d.params, toks[idx], d.cfg, d.LAYER, jnp.asarray([d.POS]),
                                     jnp.asarray(patched)[:, None, :], d.EYE)
            pred = np.asarray(logits).argmax(-1)
            hit[r] = (pred == (r < yr[idx])).sum(); cnt[r] = len(idx)
        return hit / cnt, hit.sum() / cnt.sum()

    per, tot = place(u, means)
    v = np.random.default_rng(3).normal(size=d.D); v /= np.linalg.norm(v)
    pv = H @ v
    _, tot_r = place(v, np.array([pv[xr == r].mean() for r in range(5)]))
    print("\n3. Put x at rank r by setting the one number (x really sat elsewhere; r is not y's rank):")
    print("   r                           " + "  ".join(f"{r:>6}" for r in range(5)))
    print("   answers as if x were at r   " + "  ".join(f"{a:6.3f}" for a in per))
    print(f"   overall {tot:.3f}   (same with a random direction: {tot_r:.3f})")
