"""Is the 8-dimensional causal subspace of das_complement.py confined to particular slots?

Loads the subspace saved by das_complement.py and reports how much of it lies in each of the six
slots (a basis-free measure, summing to 8), and how concentrated each learned direction is in its
biggest slot.

    python experiments/detector_followups/das_structure.py
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
K, D = 6, 64

if __name__ == "__main__":
    U = np.load(ROOT / ".cache/detect/das_slots_k8.npy")  # [384, 8], orthonormal columns
    P = U @ U.T
    per_slot = [float(np.trace(P[D * s:D * (s + 1), D * s:D * (s + 1)])) for s in range(K)]
    print("share of the 8-dim causal subspace lying in each slot (sums to 8):")
    print("  " + "  ".join(f"slot {s}: {v:.2f}" for s, v in enumerate(per_slot)))
    conc = [max(np.sum(U[D * s:D * (s + 1), j] ** 2) for s in range(K)) for j in range(U.shape[1])]
    print("per learned direction, fraction in its biggest slot: " + "  ".join(f"{c:.2f}" for c in conc))
    print("(spread evenly would be 0.17 per direction; one slot each would be 1.00)")
