"""Pre-registered test: access x question type (docs/INTERPRETATION.md).

Train two networks on a mix of "is x left of y?" and "is x at position k?" questions (position
questions need x's place in the whole line), then examine both with the detector:

  plain-pos      plain transformer: the question can always see the premises
  side-door-pos  Slot Attention bottleneck, trained with a closing side door, then continued with
                 the door shut, so the premises reach the question only through the slots

Every stage checkpoints and is skipped when already done, so the script can simply be rerun after
an interruption.

    python experiments/detector_followups/access_x_demand.py
"""

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from hyperphantasia.detect import nets  # noqa: E402

CACHE = ROOT / ".cache/detect"
MIX = 0.5
PLAIN = CACHE / "plain-pos_seed0_steps8000_curriculum.pkl"
DOOR = CACHE / "side-door-pos_seed0_steps12000_curriculum.pkl"
SHUT = CACHE / "side-door-pos-shut-continued_seed0_steps6000.pkl"


def stage(name, out, fn):
    if out.exists():
        print(f"[{name}] already done: {out.name}", flush=True)
        return
    print(f"[{name}] training", flush=True)
    t = time.time()
    cfg, params, log = fn()
    nets.save(out, cfg, params, log)
    acc = nets.accuracy(params, cfg, __import__("hyperphantasia.detect.world", fromlist=["batch"]).batch(
        __import__("numpy").random.default_rng(1000), 2000, ns=(3, 4, 5)))
    print(f"[{name}] done in {time.time() - t:.0f}s; left-of accuracy {acc:.3f}", flush=True)


def plain():
    cfg = nets.NetConfig(kind="plain", position_questions=MIX)
    params, log = nets.train(cfg, steps=8000, log_every=500, curriculum=True, checkpoint=PLAIN.with_suffix(".ckpt"))
    return cfg, params, log


def door():
    cfg = nets.side_door_config(position_questions=MIX)
    params, log = nets.train(cfg, steps=12000, log_every=500, curriculum=True, checkpoint=DOOR.with_suffix(".ckpt"))
    return cfg, params, log


def shut():
    cfg, params, _ = nets.load(DOOR)
    params, log = nets.train(cfg, steps=6000, lr=1e-3, log_every=500, init_params=params, force_side=0.0,
                             seed=1, checkpoint=SHUT.with_suffix(".ckpt"))
    return cfg, params, log


if __name__ == "__main__":
    stage("plain-pos", PLAIN, plain)
    stage("side-door-pos", DOOR, door)
    stage("side-door-pos, door shut", SHUT, shut)
    subprocess.run([sys.executable, "-u", "-m", "hyperphantasia.detect", "--kinds", "none",
                    "--out", str(ROOT / "results/access_x_demand"),
                    "--also", f"{PLAIN}=plain transformer, mixed left-of and position questions",
                    "--also", f"{SHUT}=Phantasm Slots (side door, examined shut), mixed left-of and position questions"],
                   check=True, cwd=ROOT)
