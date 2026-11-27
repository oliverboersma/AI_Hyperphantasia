"""Where does the plain transformer get the middle of the line?

Its on-demand coordinate at x's site is sharp at the ends and blurry in the middle (das_plain.py,
das_plain_coordinate.py), yet it answers middle comparisons almost perfectly. This script asks
which positions carry the information that decides the answer. Pairs are split by where x lands
in the source world (an end, or the middle) and by whether the source answer is stated in the
premises (x and y adjacent: the premises are exactly the adjacent pairs) or needs composing.

On "move only x" pairs (x swaps with a letter beyond y; y keeps its place; the answer flips),
answered correctly in both worlds, residuals from the source world are transplanted at one layer
into the base run, at: x's site, y's site, both, the premise positions, or the premises and x's
site. The share of answers that follow the source world is reported. If the middle comes from
premises read late, the premise transplant at layers 2 and 3, after x's coordinate is built, should
carry the middle cases better than the ends.

The source world's premises are written into the base world's premise slots (a world of n letters
always has n - 1 premises), so both runs share one padding mask and a transplant of every position at any layer
reproduces the source run exactly.

    python experiments/detector_followups/middle_source.py [checkpoint]

Set N_LETTERS (default 5) to use worlds of another size.
"""

import os
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, world as W  # noqa: E402

PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / ".cache/detect/plain_seed0_steps8000_curriculum.pkl"
cfg, params, _ = nets.load(PATH)
R = nets.roles(cfg)
X, Y, ANS, Q = R["X"][0], R["Y"][0], R["ANS"][0], R["scene"][0]
PREM = list(range(Q))  # BOS and the premise slots, everything before the question
EYE = jnp.eye(cfg.d)
N = int(os.environ.get("N_LETTERS", 5))  # letters per world
ENDS = [0, N - 1]


def encode_in_slots(src, base):
    """Encode the source worlds with their premises in the base worlds' slots."""
    tb, ts = nets.encode(cfg, base), nets.encode(cfg, src)
    out = tb.copy()
    for i, s in enumerate(src):
        slots = [k for k in range(W.MAX_N - 1) if tb[i, 1 + 3 * k] != W.PAD]
        assert len(slots) == len(s.premises)
        for k, prem in zip(slots, s.premises):
            out[i, 1 + 3 * k: 4 + 3 * k] = prem
        out[i, Q:] = ts[i, Q:]
    return out


if __name__ == "__main__":
    rng = np.random.default_rng(5)
    base, src = [], []
    while len(base) < 8000:
        p = W.sample(rng, N)
        s = W.rank_counterfactual(p, rng)
        if s is not None:
            base.append(p); src.append(s)
    tb = jnp.asarray(nets.encode(cfg, base))
    ts = jnp.asarray(encode_in_slots(src, base))
    lb, rb = nets.forward(params, tb, cfg)
    ls, rs = nets.forward(params, ts, cfg)
    target = W.labels(src)
    ok = (np.asarray(lb).argmax(-1) == W.labels(base)) & (np.asarray(ls).argmax(-1) == target)

    rs_x = np.array([s.rank[s.query[0]] for s in src])
    ry = np.array([p.rank[p.query[1]] for p in base])
    case = np.char.add(np.where(np.isin(rs_x, ENDS), "end, ", "middle, "),
                       np.where(np.abs(rs_x - ry) == 1, "stated", "composed"))
    names = ("end, stated", "end, composed", "middle, stated", "middle, composed")
    print(f"{PATH.name}: {ok.sum()} of {len(base)} pairs answered correctly in both worlds")
    print("  where x lands in the source world, and whether its answer is stated or composed")
    print("  pairs per case: " + ", ".join(f"{c} {int((ok & (case == c)).sum())}" for c in names))
    acc = np.asarray(ls).argmax(-1) == target
    print("  network accuracy on the source worlds, by case: "
          + ", ".join(f"{c} {acc[case == c].mean():.3f}" for c in names))

    sites = [("x's site", [X]), ("y's site", [Y]), ("x and y", [X, Y]), ("premises", PREM),
             ("premises and x", PREM + [X]), ("everything", list(range(tb.shape[1])))]
    print("\nShare of answers that follow the source world after the transplant:")
    print(f"  {'layer':<6} {'transplant':<16}" + "".join(f"{c:>18}" for c in names))
    for layer in (1, 2, 3):
        for name, pos in sites:
            vals = rs[layer][:, jnp.asarray(pos)]
            lg, _ = nets.forward(params, tb, cfg, layer, jnp.asarray(pos), vals, EYE)
            follows = np.asarray(lg).argmax(-1) == target
            print(f"  {layer:<6} {name:<16}" + "".join(f"{follows[ok & (case == c)].mean():>18.3f}" for c in names),
                  flush=True)

    # Plain accuracy on fresh questions, by type (no transplants)
    rng = np.random.default_rng(77)
    probs = [W.sample(rng, N) for _ in range(20000)]
    pred = np.asarray(nets.forward(params, jnp.asarray(nets.encode(cfg, probs)), cfg)[0]).argmax(-1)
    right = pred == W.labels(probs)
    rx = np.array([p.rank[p.query[0]] for p in probs]); ryy = np.array([p.rank[p.query[1]] for p in probs])
    gap = np.abs(rx - ryy)
    stated = gap == 1
    end = np.isin(rx, ENDS) | np.isin(ryy, ENDS)
    print(f"\nAccuracy on fresh {N}-letter questions (no transplant):")
    rows = [("stated pair", stated), ("composed, a letter at an end", ~stated & end),
            ("composed, both letters in the middle", ~stated & ~end)]
    rows += [(f"  ... of which {g} apart", ~stated & ~end & (gap == g)) for g in range(2, N - 2)]
    for name, m in rows:
        if m.sum():
            print(f"  {name:<40} {right[m].mean():.3f}  ({int(m.sum())} questions)")
