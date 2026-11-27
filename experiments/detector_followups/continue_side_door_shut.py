"""Continue training the side-door network for 6000 steps with the door kept shut (lr 1e-3).

Starts from the network trained by `train(side_door_config(), steps=12000, curriculum=True)`, whose
shut-door accuracy was still rising (77%) when its learning rate ran out. Output from our run:
continue_side_door_shut.out.txt (77% -> 95% held-out accuracy with the door shut).

    python experiments/detector_followups/continue_side_door_shut.py
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets  # noqa: E402

src = ROOT / ".cache/detect/side-door_seed0_steps12000_curriculum.pkl"
out = ROOT / ".cache/detect/side-door-shut-continued_seed0_steps6000.pkl"

if __name__ == "__main__":
    cfg, params, _ = nets.load(src)
    t = time.time()
    params, log = nets.train(cfg, steps=6000, lr=1e-3, log_every=250, init_params=params, force_side=0.0,
                             seed=1, checkpoint=out.with_suffix(".ckpt"))
    nets.save(out, cfg, params, log)
    print("continued done in", round(time.time() - t), "s")
