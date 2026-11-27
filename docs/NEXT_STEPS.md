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

**Status: done.** With the door shut the network reached 95%. Its slots carry everything the answer
needs (90% transplant transfer), but neither positions nor premises are linearly readable from
them, and composition still happens at the question. Next: a non-linear probe or learned rotation
on the slots (item 4), and dense questions (item 5), which should be the case where a slot scene
pays.

## 1b. Is the non-linear scene in the slots causally used?

The bottleneck trained on mixed questions holds ranks in its slots in a non-linear code (MLP probe:
0.83, against 0.35 for the raw premises). Learn a rotation of the slot space that isolates rank (a
distributed alignment search), or use the MLP probe's gradients to edit rank, and check whether
the answer follows the edited positions.

**Settles.** Whether this is a scene the network computes from, or information that is merely
present.

**Status: done** (`scene_edit.py`). Partly used: probe-guided edits flip the answer 22% of the time,
against 7% for a random edit of the same size and 100% for the real slots of that world.

**Follow-up: done** (`das_slots.py`, `das_complement.py`, `das_structure.py`). Distributed alignment
search found that 8 of the 384 slot dimensions carry the answer (83%, against 86% for all of them
and 17% for random directions). The readable scene sits in the other 376 dimensions, and
transplanting those alone moves the answer only 19% of the time. The causal code is a compact
ordering code spread across every slot, and the readable scene beside it is mostly idle.

**Open.** Is the idle copy a leftover of the side-door schedule? Run the same search on a second
seed and on the left-of-only bottleneck, and check whether pruning the idle 376 dimensions costs
any accuracy.

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

  **Status: done for seed 0** (`das_plain.py`, `das_plain_coordinate.py`). The code is mostly one
  number, x's place on the line (one direction carries 80% of the site's 82%). It is a coordinate
  rather than a set of relations (73–81% against 41–46%), sharp at the ends and blurry in the middle.

  **Seeds 1 and 2: done.** Both hold a coordinate too. Their main directions resolve less of the line:
  seed 1 both ends, seed 2 one end. Those can be read from surface counts without composing, so
  only seed 0's main direction goes beyond them. The order matches the composition scores.

  **Position-question network: done** (`das_plain_pos.py`). One number line serves both question
  types: the directions used for each nearly coincide, and the left-of-learned one answers position
  questions. The expected change did not happen. The main direction still only marks the ends; the
  middle lives on a second, weaker axis.

  Next:
  * **The middle's source.** Find where the networks get the finer middle information that their
    coordinates lack, perhaps the stated premises read at y's site (a transplant at y's site, and
    patching the premise tokens).
  * **Seeds for the position network.** Only one was trained.
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
