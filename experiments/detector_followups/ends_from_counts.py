"""Can the ends of the line be found without composing premises?

In every 5-letter world, is the first letter the only one never stated to be right of anything,
and the last the only one never stated to be left of anything? If so, an "end detector" at x's
site needs only surface counts.

    python experiments/detector_followups/ends_from_counts.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from hyperphantasia.detect import world as W  # noqa: E402

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    first = last = 0
    n = 5000
    for _ in range(n):
        p = W.sample(rng, 5)
        f = W.surface_features(p)
        first += [L for L in p.order if f[L][1] == 0] == [p.order[0]]
        last += [L for L in p.order if f[L][0] == 0] == [p.order[-1]]
    print(f"first letter is the only one never stated right of anything: {first / n:.3f}")
    print(f"last letter is the only one never stated left of anything:  {last / n:.3f}")
