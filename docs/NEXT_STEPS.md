# Next steps

What to do after the first round of imagination-detector experiments
([IMAGINATION_DETECTOR.md](IMAGINATION_DETECTOR.md)), in priority order. Each item says what it
would settle.

## 1. Get a network with a forced standing scene to learn

**Why.** The architecture question is still open, because no slot-bottleneck network has learned
the task. Four slot designs stayed at chance, while the same encoder and decoder without the
bottleneck learned normally.

**Plan: a closing side door.** Give the decoder a gated extra route straight to the encoded
premises. The gate is fully open for the first 40% of training, closes gradually until 80%, and stays shut
for the last fifth. The network learns the task the easy way, then has to move what it knows into the
slots. Evaluation always runs with the gate shut, so it is a pure bottleneck network.
(`slot_side_door` in `hyperphantasia/detect/nets.py`; status in section 6 of the detector doc.)

**Settles.** If it works, the detector can finally ask whether a scene the architecture forces is a
real scene (decodable ranks, causally used), or only copies of the premises.

## 2. Compare forced scenes with on-demand codes

Once item 1 yields a working network, fit a rotation (as in `rotation_transfer.py`) between its
slots or question-site code and the plain transformers' on-demand code.

**Settles.** Whether imagination forced by the architecture uses the same geometry that training
invents on its own.

## 3. Robustness

* **Seeds.** Two more runs each of the bottleneck designs, the no-bottleneck control, and the
  side-door network.
* **Report spread, not single numbers.** Turn the per-seed tables into means and ranges.

**Settles.** Whether "building the picture first is hard to learn" and "codes are shared across
runs" hold beyond one run per design.

## 4. Read the on-demand code properly

* **Learned rotation.** Use distributed alignment search: learn a rotation at x's site that isolates
  one causal variable, instead of relying on probe directions.
* **Candidate variables.** Test "rank of x", "letters left of x", and "distance to the nearest end".

**Settles.** What exactly the plain transformer computes when it is asked, which is so far only
partly characterised.

## 5. Harder and more varied worlds

* **Other structures.** 2-D grids, trees (family relations), and processes with state.
* **Longer worlds.** More letters than were ever seen in training.
* **Mixed questions.** Several question types about the same premises, which should favour a
  standing scene, since one scene can answer them all.

**Settles.** Whether on-demand composition is a quirk of yes/no ordering questions or a general
preference of trained networks.

## 6. Scale up

* **Hardware.** A GPU, larger models (more layers and width), and 10–100× longer training.
* **Real language models.** Apply the battery to an open-weights LLM through activation hooks
  (section 9 of the detector doc). From this environment that needs PyTorch and model downloads,
  which the network policy currently blocks.

**Settles.** Whether the small-network findings say anything about large models.

## 7. Feed results back into the architecture

Update the Phantasm Slots spec in [ARCHITECTURE.md](ARCHITECTURE.md) with what was learned:

* **Learnability.** A bottleneck written before the question is hard to learn.
* **The side door.** If item 1 works, the side-door schedule helps.
* **Shared codes.** On-demand codes are shared across runs.

The hyperphantasia scaffold could also try the cheap version of the finding: build only the part of
the scene that a question needs.
