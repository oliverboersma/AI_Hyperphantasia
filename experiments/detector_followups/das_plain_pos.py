"""Alignment search on the plain transformer trained with position questions (docs/NEXT_STEPS.md, item 4).

The network (plain-pos, seed 0) was trained on a mix of "is x left of y?" and "is x at position
k?". Position questions need x's place in the whole line, middle included, so its on-demand code at
x's site should resolve the middle better than the left-of-only networks' (das_plain.py). Here
the network gets the premises, a dummy left-of question, then the target question, as in
training. The site is x in the target question, at layer 2.

A. Size: interchange accuracy of learned subspaces on "move only x" left-of pairs.
B. Coordinate or relations, on left-of questions about a reordered world (as in das_plain.py).
C. One number line for both questions? Position-question pairs: the source world puts x at another
   rank, and the answer should follow the source. Test the subspace learned on LEFT-OF questions
   only, a subspace learned on position questions, and the overlap between the two.
D. The main direction by x's rank, and its relation to rank beyond surface counts.
E. Placement. Set x's code to its mean value for rank r. Then, for left-of questions, the answer
   should change as if x sat at r. For position questions, "is x at position r?" should become
   yes, and "is x at its real position?" should become no.

    python experiments/detector_followups/das_plain_pos.py
"""

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, world as W  # noqa: E402

PATH = ROOT / ".cache/detect/plain-pos_seed0_steps8000_curriculum.pkl"
LAYER = 2  # the best x-site layer for this network too (results/access_x_demand)
cfg, params, _ = nets.load(PATH)
D = cfg.d
EYE = jnp.eye(D)
POS = 1 + 3 * (W.MAX_N - 1) + 4 + 1  # x in the second question block


def tokens(probs, qs, dummies):
    return jnp.asarray(nets._slotted(probs, cfg, True, [[d, q] for d, q in zip(dummies, qs)]))


def site(probs, qs, dummies):
    toks = tokens(probs, qs, dummies)
    logits, resid = nets.forward(params, toks, cfg)
    return toks, resid[LAYER][:, POS], np.asarray(logits).argmax(-1)


def run(toks, hb, hs, U):
    patched = hb + ((hs - hb) @ U) @ U.T
    logits, _ = nets.forward(params, toks, cfg, LAYER, jnp.asarray([POS]), patched[:, None, :], EYE)
    return logits


def frame(Wm):
    q, _ = jnp.linalg.qr(Wm)
    return q


def loss_fn(Wm, toks, hb, hs, y, w):
    nll = -jnp.take_along_axis(jax.nn.log_softmax(run(toks, hb, hs, frame(Wm))), y[:, None], 1)[:, 0]
    return (nll * w).sum() / w.sum()


grad_fn = jax.jit(jax.value_and_grad(loss_fn))


def reorder(p, rng, x=None):
    while True:
        order = tuple(int(v) for v in rng.permutation(p.order))
        if x is None or order.index(x) != p.rank[x]:
            return W.Problem(order, W.premises_for(order, rng), p.query)


def pairs(rng, size, kind):
    """kind 'move_x' or 'reorder' (left-of questions), or 'pos' (position questions)."""
    base, src, qs, dummies = [], [], [], []
    rel, coord, valid, b_ans = [], [], [], []
    while len(base) < size:
        p = W.sample(rng, 5)
        x, y = p.query
        if kind == "move_x":
            s = W.rank_counterfactual(p, rng)
            if s is None:
                continue
        else:
            s = reorder(p, rng, x if kind == "pos" else None)
        if kind == "pos":
            k = s.rank[x] if rng.random() < 0.5 else p.rank[x]
            q = ("pos", x, k)
            a = int(s.rank[x] == k)
            rel.append(a); coord.append(a); valid.append(True); b_ans.append(int(p.rank[x] == k))
        else:
            q = (x, y)
            rel.append(s.answer); coord.append(int(s.rank[x] < p.rank[y]))
            valid.append(s.rank[x] != p.rank[y]); b_ans.append(p.answer)
        base.append(p); src.append(s); qs.append(q)
        dummies.append(tuple(int(v) for v in rng.choice(p.order, size=2, replace=False)))
    tg = {"relations": np.array(rel), "coordinate": np.array(coord), "valid": np.array(valid),
          "base": np.array(b_ans)}
    return base, src, qs, dummies, tg


def train(k, kind, hyp="relations", steps=800, seed=0):
    rng = np.random.default_rng(seed)
    Wm = jnp.asarray(np.random.default_rng(seed + 1).normal(size=(D, k)) / np.sqrt(D))
    m, v = jnp.zeros_like(Wm), jnp.zeros_like(Wm)
    for t in range(1, steps + 1):
        base, src, qs, dm, tg = pairs(rng, 256, kind)
        toks, hb, _ = site(base, qs, dm)
        _, hs, _ = site(src, qs, dm)
        _, g = grad_fn(Wm, toks, hb, hs, jnp.asarray(tg[hyp]), jnp.asarray(tg["valid"], jnp.float32))
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        Wm = Wm - 1e-2 * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8)
    return frame(Wm)


EVAL = {kind: pairs(np.random.default_rng(99 + i), 3000, kind) for i, kind in enumerate(("move_x", "reorder", "pos"))}
CACHE = {}


def iia(kind, U, hyp="relations"):
    base, src, qs, dm, tg = EVAL[kind]
    if kind not in CACHE:
        toks, hb, _ = site(base, qs, dm)
        CACHE[kind] = (toks, hb, site(src, qs, dm)[1])
    toks, hb, hs = CACHE[kind]
    pred = np.asarray(run(toks, hb, hs, U)).argmax(-1)
    keep = tg["valid"] & (tg[hyp] != tg["base"])
    return float((pred == tg[hyp])[keep].mean())


def rand(k, seed):
    return frame(jnp.asarray(np.random.default_rng(seed).normal(size=(D, k))))


def overlap(U, V):
    """Mean squared cosine of the principal angles between two subspaces (1 = same, ~k/D = unrelated)."""
    s = np.linalg.svd(np.asarray(U).T @ np.asarray(V), compute_uv=False)
    return float((s ** 2).mean())


if __name__ == "__main__":
    print(f"{PATH.name}, x's site in the second question, layer {LAYER}")
    print("\nA. Left-of, move only x (held-out pairs whose answers differ). Answer follows the source:")
    print(f"  whole site (64 dims)   {iia('move_x', EYE):.3f}")
    lo = {}
    for k in (1, 2, 4, 8):
        lo[k] = train(k, "move_x")
        print(f"  k={k:<3} learned {iia('move_x', lo[k]):.3f}   random {iia('move_x', rand(k, k)):.3f}", flush=True)

    print("\nB. Left-of, reordered world. Each hypothesis scored where its answer differs from the base:")
    print(f"  {'':<26} {'relations':>10} {'coordinate':>11}")
    for name, U in [("whole site", EYE), ("learned (left-of), k=1", lo[1]), ("learned (left-of), k=8", lo[8])]:
        print(f"  {name:<26} {iia('reorder', U, 'relations'):>10.3f} {iia('reorder', U, 'coordinate'):>11.3f}",
              flush=True)

    print("\nC. Position questions ('is x at position k?'); the source world puts x elsewhere.")
    print("   Answer follows the source:")
    print(f"  whole site                              {iia('pos', EYE):.3f}")
    po = {k: train(k, "pos", seed=3) for k in (1, 2)}
    for k in (1, 2):
        print(f"  k={k}: learned on LEFT-OF only {iia('pos', lo[k]):.3f}   learned on position questions "
              f"{iia('pos', po[k]):.3f}   random {iia('pos', rand(k, 10 + k)):.3f}", flush=True)
        print(f"       and the position-learned subspace on left-of move-x pairs: {iia('move_x', po[k]):.3f}")
    for k in (1, 2):
        print(f"  overlap of the two learned subspaces, k={k}: {overlap(lo[k], po[k]):.3f}   "
              f"(random pair: {overlap(rand(k, 20 + k), rand(k, 30 + k)):.3f})")

    rng = np.random.default_rng(41)
    probs = [W.sample(rng, 5) for _ in range(6000)]
    xs = [int(rng.choice(p.order)) for p in probs]
    dm = [tuple(int(v) for v in rng.choice(p.order, size=2, replace=False)) for p in probs]
    xr = np.array([p.rank[x] for p, x in zip(probs, xs)])
    lo_q = [(x, int(rng.choice([z for z in p.order if z != x]))) for p, x in zip(probs, xs)]
    pos_q = [("pos", x, 0) for x in xs]  # x's site comes before the digit, so k does not matter here
    sites = {"left-of": site(probs, lo_q, dm), "position": site(probs, pos_q, dm)}
    u = np.asarray(lo[1])[:, 0]

    print("\nD. The left-of-learned direction by x's rank, at each question type's x site:")
    print("   rank           " + "  ".join(f"{r:>6}" for r in range(5)))
    for name, (_, H, _) in sites.items():
        proj = np.asarray(H) @ u
        print(f"   {name:<14} " + "  ".join(f"{proj[xr == r].mean():6.2f}" for r in range(5)))
    F = np.array([W.surface_features(p)[x] for p, x in zip(probs, xs)])
    F1 = np.c_[F, np.ones(len(F))]
    res = lambda v: v - F1 @ np.linalg.lstsq(F1, v, rcond=None)[0]  # noqa: E731
    proj = np.asarray(sites["left-of"][1]) @ u
    rho = np.corrcoef(np.argsort(np.argsort(proj)), xr)[0, 1]
    print(f"   rank correlation {rho:.3f}; beyond surface counts {np.corrcoef(res(proj), res(xr * 1.0))[0, 1]:.3f}")

    def place_left_of(U):
        toks, H, unp = sites["left-of"]
        H = np.asarray(H); Z = H @ U
        M = np.stack([Z[xr == r].mean(0) for r in range(5)])
        yr = np.array([p.rank[q[1]] for p, q in zip(probs, lo_q)])
        hit, cnt = np.zeros(5), np.zeros(5)
        for r in range(5):
            idx = np.where((yr != r) & (xr != r))[0]
            want = (r < yr[idx]).astype(int)
            idx, want = idx[want != unp[idx]], want[want != unp[idx]]
            patched = H[idx] + (M[r] - H[idx] @ U) @ U.T
            lg, _ = nets.forward(params, toks[idx], cfg, LAYER, jnp.asarray([POS]), jnp.asarray(patched)[:, None, :], EYE)
            hit[r] = (np.asarray(lg).argmax(-1) == want).sum(); cnt[r] = len(idx)
        return hit / cnt, hit.sum() / cnt.sum()

    def place_position(U):
        """x's code set to rank r's mean (at position-question sites). Ask 'is x at r?' (want yes)
        and 'is x at its real position?' (want no)."""
        _, H, _ = sites["position"]
        H = np.asarray(H); Z = H @ U
        M = np.stack([Z[xr == r].mean(0) for r in range(5)])
        out = {"yes at r": np.zeros(5), "no at its real place": np.zeros(5)}
        for r in range(5):
            idx = np.where(xr != r)[0]
            patched = jnp.asarray(H[idx] + (M[r] - H[idx] @ U) @ U.T)[:, None, :]
            for name, ks, want in (("yes at r", [r] * len(idx), 1), ("no at its real place", xr[idx], 0)):
                qs = [("pos", xs[i], int(k)) for i, k in zip(idx, ks)]
                toks = tokens([probs[i] for i in idx], qs, [dm[i] for i in idx])
                unp = np.asarray(nets.forward(params, toks, cfg)[0]).argmax(-1)
                lg, _ = nets.forward(params, toks, cfg, LAYER, jnp.asarray([POS]), patched, EYE)
                pred = np.asarray(lg).argmax(-1)
                must = unp != want  # only cases where the answer has to change
                out[name][r] = (pred == want)[must].mean()
        return out

    print("\nE. Placement: set x's code to its mean for rank r (cases where the answer must change)")
    print("   left-of questions, answer changes as if x sat at r:")
    print("   subspace                     " + "  ".join(f"r={r:<4}" for r in range(5)) + "  overall")
    for name, U in [("left-of-learned, k=1", lo[1]), ("left-of-learned, k=2", lo[2]),
                    ("random direction", rand(1, 5))]:
        per, tot = place_left_of(np.asarray(U))
        print(f"   {name:<28} " + "  ".join(f"{a:6.3f}" for a in per) + f"  {tot:.3f}", flush=True)
    print("   position questions:")
    for name, U in [("left-of-learned, k=1", lo[1]), ("left-of-learned, k=2", lo[2]),
                    ("position-learned, k=2", po[2]), ("random direction", rand(1, 5))]:
        res_ = place_position(np.asarray(U))
        for q, per in res_.items():
            print(f"   {name:<22} {q:<21}" + "  ".join(f"{a:6.3f}" for a in per) + f"  {per.mean():.3f}", flush=True)
