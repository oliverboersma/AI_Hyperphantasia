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

    # surface comparison: the best linear guess of x's rank from counts a reader needs no composing for
    F = np.array([W.surface_features(p)[p.query[0]] for p in probs])
    F1 = np.c_[F, np.ones(len(F))]
    guess = F1 @ np.linalg.lstsq(F1, xr, rcond=None)[0]
    rho_s = np.corrcoef(np.argsort(np.argsort(guess)), xr)[0, 1]
    res = lambda v: v - F1 @ np.linalg.lstsq(F1, v, rcond=None)[0]  # noqa: E731
    partial = np.corrcoef(res(proj), res(xr.astype(float)))[0, 1]
    print(f"   surface counts' best linear guess of the rank: rank correlation {rho_s:.3f}")
    print(f"   correlation with x's rank once surface counts are partialled out: {partial:.3f}")

    yr = np.array([p.rank[p.query[1]] for p in probs])
    unpatched = np.asarray(nets.forward(d.params, toks, d.cfg)[0]).argmax(-1)

    def place(direction, levels):
        """Set x's coordinate along `direction` to levels[r]. Score 'answer == (r < rank(y))' on the
        cases where that differs from what the unpatched network says, i.e. the answer must change."""
        hit, cnt = np.zeros(5), np.zeros(5)
        for r in range(5):
            idx = np.where((yr != r) & (xr != r))[0]  # x actually moves, and r is not y's spot
            want = (r < yr[idx]).astype(int)
            idx, want = idx[want != unpatched[idx]], want[want != unpatched[idx]]
            hb = H[idx]
            patched = hb + np.outer(levels[r] - hb @ direction, direction)
            logits, _ = nets.forward(d.params, toks[idx], d.cfg, d.LAYER, jnp.asarray([d.POS]),
                                     jnp.asarray(patched)[:, None, :], d.EYE)
            hit[r] = (np.asarray(logits).argmax(-1) == want).sum(); cnt[r] = len(idx)
        return hit / np.maximum(cnt, 1), hit.sum() / cnt.sum(), cnt

    per, tot, cnt = place(u, means)
    v = np.random.default_rng(3).normal(size=d.D); v /= np.linalg.norm(v)
    pv = H @ v
    per_r, tot_r, _ = place(v, np.array([pv[xr == r].mean() for r in range(5)]))
    print("\n3. Put x at rank r by setting the one number, on cases where that must change the answer:")
    print("   r                           " + "  ".join(f"{r:>6}" for r in range(5)))
    print("   cases                       " + "  ".join(f"{int(c):>6}" for c in cnt))
    print("   answer changes as it should " + "  ".join(f"{a:6.3f}" for a in per))
    print("   same, random direction      " + "  ".join(f"{a:6.3f}" for a in per_r))
    print(f"   overall {tot:.3f}   (random direction {tot_r:.3f})")
