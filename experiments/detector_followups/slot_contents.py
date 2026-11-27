"""What do the trained bottleneck network's slots hold: the premises, or a scene?

Linear probes on the six slot vectors (decoder layer 0, flattened), compared with an untrained
copy of the same network:

* premise content: for each letter L, which letter sits directly right of L (one of the other 25,
  or none). This is stated in a single premise.
* scene content: L's rank (position 0-4 in the line). This needs the premises composed.

    python experiments/detector_followups/slot_contents.py
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets, probes, world as W  # noqa: E402
from hyperphantasia.detect.subjects import NeuralSubject  # noqa: E402

cfg, params, _ = nets.load(ROOT / ".cache/detect/side-door-shut-continued_seed0_steps6000.pkl")
subjects = {"trained": NeuralSubject(cfg, params, "trained"),
            "untrained": NeuralSubject(cfg, nets.init(cfg, 99), "untrained")}
rng = np.random.default_rng(21)
probs = [W.sample(rng, 5) for _ in range(8000)]


def per_letter(acts, target, classes):
    scores = []
    for L in range(W.LETTERS):
        idx = [i for i, p in enumerate(probs) if L in p.rank]
        y = np.array([target(probs[i], L) for i in idx])
        scores.append((probes.fit(acts[idx], y, classes, seed=L).acc, len(idx)))
    return np.average([a for a, _ in scores], weights=[n for _, n in scores])


def successor(p, L):
    r = p.rank[L]
    return p.order[r + 1] if r + 1 < p.n else W.LETTERS  # 26 = none (L is rightmost)


if __name__ == "__main__":
    print("probe accuracy on the slots (decoder layer 0), per letter, held-out:")
    for name, subj in subjects.items():
        _, acts = subj.run(probs)
        A = acts[(0, "scene")].reshape(len(probs), -1)
        succ = per_letter(A, successor, W.LETTERS + 1)
        rank = per_letter(A, lambda p, L: p.rank[L], 5)
        print(f"  {name:<9}  premise content (right neighbour): {succ:.2f}   scene content (rank): {rank:.2f}")
    print("chance: right neighbour about 0.20 (always guess 'none'), rank 0.20")
