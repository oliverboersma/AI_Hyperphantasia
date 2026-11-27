# Does imagination exist in the architecture, the mechanism, or neither?

When a network answers a question that requires picturing a situation, three explanations are
possible:

1. **The architecture supplies an imagination.** The design gives it a place to build a model of
   the situation, and it uses that place.
2. **The mechanism invented one.** Nothing in the design asks for a model, but training produced
   one inside the weights.
3. **Something else.** It gets the answers without any model of the situation: surface
   statistics, memorised patterns, or a direct computation of the relation asked about.

This document describes an instrument that tells these apart, and what it found. Run it with:

```bash
pip install -e ".[detect]"
```

```bash
python -m hyperphantasia.detect
```

The code is in `hyperphantasia/detect/`. The report is written to
`results/imagination_report.md`.

---

## 1. Making the question testable

"Imagination" here means something specific. A system imagines a situation if all of these hold:

| # | Criterion | How it is tested |
|---|---|---|
| 1 | It can answer questions whose answers are **not stated** but follow from putting the premises together | Behaviour on questions that need composition |
| 2 | Somewhere inside, there is a **representation of the situation's state** | Linear probes for that state, compared against controls |
| 3 | The answer is **produced from** that representation | Interchange interventions: transplant the representation from a different situation and see whether the answer follows it |
| 4 | *(timing)* The representation is **standing**: built from the premises before the question is known, so any question could be answered from it | The same probes and transplants at the site that comes after the premises but before the question |

Criterion 3 is what separates imagination from information that is merely present. Probes alone
cannot do this, because a probe can decode information the network never uses. That is a
well-known weakness of probing (Belinkov 2022; Ravichander et al. 2021). The detector therefore
requires causal evidence.

The verdicts, from strongest to weakest:

| Verdict | Meaning |
|---|---|
| **standing scene** | Imagination in the mechanism: a model of the whole scene exists before the question and the answer comes from it |
| **on-demand scene** | No standing model, but when the question arrives, the positions of the asked-about items are built and used |
| **epiphenomenal** | Scene information can be decoded, but editing it does not change the answer |
| **something else** | Correct answers, with no scene representation this detector can find |
| **no imagination** | Fails questions that need composition: stated pairs plus surface statistics |

Thresholds are fixed in `detect/verdict.py`, chosen before any trained network was examined.

## 2. The test bed: hidden-order worlds

Each problem has a secret left-to-right order of letters. The subject sees only the neighbouring
pairs, shuffled, each written either as `a < b` or as `b > a`, then a question:

```
B > K ; F < B ; K < Q ; Q < D ? F D =        world: F B K Q D      answer: yes (F is left of D)
```

This task has three useful properties.

* **Composition is required.** A question about two letters that never share a premise can only
  be answered by chaining premises together.
* **Surface statistics are not enough.** Counting how often a letter is on the "smaller" side
  finds the two ends of the line and nothing else. All middle letters look identical
  (`tests/test_detect.py::test_surface_counts_cannot_place_the_middle`).
* **The state is crisp and editable.** Each letter's rank is the scene. Counterfactual worlds can
  be generated that change exactly one thing:
  * **scene counterfactual**: same letters, same question, a different order in which the answer
    flips;
  * **rank counterfactual**: only the first asked-about letter moves (it swaps with a letter on
    the far side of the second one). The second letter keeps its rank, and the answer flips.

## 3. The battery

Every subject exposes its activations at four **sites** on every layer:

* **scene**: after the premises, before the question;
* **X** and **Y**: the positions of the two asked-about letters;
* **ANS**: where the answer is produced.

The detector may overwrite (part of) a site and let the computation continue.

1. **Behaviour.** Accuracy overall, on questions about stated pairs, and on questions that need
   composition. Consistency of answers across four reshuffles of the same world. Accuracy on
   7-letter worlds, which are never seen in training (reported, not used in the verdict).
2. **Representation.** Ridge-regression probes for rank, with the regularisation chosen on held-out
   data and accuracy reported on a third split.
   * At X and Y, one probe decodes the asked-about letter's rank.
   * At *scene*, one probe per letter decodes every letter's rank (the whole scene).
   * Each result is compared with two controls: **the same probe on an untrained copy of the same
     network** (what architecture and input alone give), and **the same probe on surface counts**
     (what a reader could know without composing). These follow the control-task logic of Hewitt
     & Liang (2019).
3. **Causation.**
   * *Interchange intervention* (causal tracing; Meng et al. 2022, Geiger et al. 2021). For every
     layer and site, transplant the whole activation from the scene counterfactual and measure
     how often the answer follows the transplanted world.
   * *Subspace interchange*, in the spirit of distributed alignment search (Geiger et al. 2023).
     At X, transplant **only the directions the rank probe reads**, using the rank
     counterfactual. If the network really answers by comparing "rank of x" with "rank of y",
     moving x's rank alone must flip the answer. The control is a random subspace of the same
     size.
4. **Timing.** Comes from the scene site. If the whole scene is decodable there *and*
   transplanting it moves the answer, the model is standing. Otherwise any scene that exists is
   built after the question arrives.

## 4. Calibrating the instrument

A detector is only trustworthy if it gets the right answer on cases where the truth is known.
The command refuses to examine any network until both hand-built subjects pass:

| Subject | Internals (known by construction) | Verdict |
|---|---|---|
| **Scene engine** | Builds every letter's rank first, then reads the answer off it | **standing scene** ✓ |
| **Surface heuristic** | Looks up stated pairs; otherwise compares how often each letter is on the "smaller" side; ties go to mention order | **no imagination** ✓ |

The surface heuristic is a useful trap. Its scene site is a lossless copy of the premises, so
transplanting that site *does* move its answers (95%). A detector that relied on transplants
alone would call it imaginative. The ranks, though, cannot be decoded there beyond what surface
counts give, and its behaviour on composition is at chance (50%). The verdict correctly says
no.

## 5. Subjects

Every network is trained **only on yes/no answers**. Nothing about the hidden order is ever
supervised. To get more answer signal per pass, each training sequence asks four questions about
the same premises. Evaluation asks one, and because attention only looks backwards, later
questions cannot affect the first answer.

**Input layout.** There is a fixed grid of six premise slots with learned absolute positions.
Each premise goes into a randomly chosen slot, and unused slots are masked out. A premise's own
three tokens stay adjacent, which makes them easy to group, but no slot is special. Training
starts with small worlds (two or three letters) and grows to five. Getting here took several
failed attempts, which are instructive:

* **Premises packed against the question**, with the same curriculum. The plain transformer learned
  to read fixed positions just before the question. It was perfect on 2–3 letter worlds, but on
  5-letter worlds it answered only 65% of questions about pairs that were stated outright. This
  network is examined as an extra subject.
* **Packed premises, all sizes mixed from the start.** No progress in 2000 steps.
* **No positions, but random shared premise IDs and role tags.** Even two-letter worlds were not
  learned in a short run, where absolute positions learned them in 300 steps: grouping a premise's
  tokens by ID match is itself hard to discover.

The subjects:

* **Plain transformer**: 4 layers, width 64, 4 heads, causal attention over the whole sequence.
  Any imagination it has is an invention of its mechanism.
* **Phantasm Slots transformer**: a 3-layer encoder (attending in both directions) writes the
  premises into slot vectors, and a 4-layer decoder answers the question while able to see *only*
  the slots. Four ways of filling the slots were tried: 8 free slot vectors after the premises;
  6 slot vectors each anchored right after one premise slot; 6 learned slots that compete for
  premise tokens (Slot Attention; Locatello et al., 2020); and Slot Attention that also writes each
  premise letter's raw embedding into the slots. The architecture forces any scene through the
  bottleneck.

## 6. Results

One seed, trained on CPU. Full tables are in `results/imagination_report.md`. Calibration passed
before any network was examined.

| Subject | Composition questions | Verdict |
|---|---|---|
| Scene engine (known positive) | 1.00 | **standing scene** ✓ |
| Surface heuristic (known negative) | 0.50 | **no imagination** ✓ |
| Plain transformer, packed premises | 0.58 | **no imagination** (positional shortcut) |
| Plain transformer, slotted premises | 0.905 | **something else** |
| Phantasm Slots, four slot designs | not learned | not examined: all at chance (see below) |

**The working plain transformer** gets 98.9% of 5-letter questions right overall, 99.6% on stated
pairs, and 86% on 7-letter worlds it never saw in training. Its answers are the same under
reshuffling 98% of the time. Inside:

* **No standing scene.** At the site after the premises and before the question, transplanting
  activations from a world with the opposite answer never moves the answer (1% at every layer).
  The rank probe there (0.37–0.41) reads less than surface counts do (0.60).
* **Built on demand, at the asked-about letters.** At layer 2, transplanting x's site moves the
  answer 42% of the time, and y's site 61%. When only x moves (y keeps its place), transplanting
  x's site moves the answer 81% of the time. By layer 3 the answer sits at the answer position
  (99%).
* **Not a coordinate.** x's rank is only partly readable at its site (0.69, against 0.59 from surface
  counts and 0.20 from an untrained copy). Transplanting just the rank directions does nothing
  (2%, the same as random directions).
* **Exploratory follow-up** (post hoc, `experiments/detector_followups/`):
  * **Location of the code.** Most of what matters at x's site lies in the space read by "is letter
    L left of x?" probes. Transplanting those 26 directions moves the answer 78% of the time,
    against 38% for a random space of the same size.
  * **Not one switch per letter.** Editing the direction for y alone works no better than editing
    an uninvolved letter's (18% vs 17%).
  * **Only partly readable.** The relation is 78% decodable there, below a surface baseline (96%).

**Three seeds.** Two more plain transformers were trained from different random starts (seeds 1
and 2; reports in `results/seed1/` and `results/seed2/`). They reach 96% on held-out worlds, but
only 83% and 73% on questions that need composition. That is below the 90% bar, so the fixed rule
calls them "no imagination", while seed 0 cleared it at 90.5%.

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Composition questions | 90.5% | 83% | 73% |
| Transplant before the question (any layer; baseline in brackets) | 1% (1%) | 7–8% (8%) | 9–17% (9%) |
| Transplant x's whole site, only x moved (best layer) | 81% | 87% | 59% |
| Transplant only the rank-probe directions (random in brackets) | 2% (3%) | 83% (12%) | 19% (19%) |

The baseline is the share of answers that already match the transplanted world's answer because
the network got that puzzle wrong.

* **Robust: no seed builds a standing scene.** In every seed, transplanting anything before the
  question barely moves the answer beyond that seed's baseline. Each one decides at the
  asked-about letters.
* **Not robust: the form of the on-demand code.** In seed 1, a few rank-probe directions at x's site
  are causally decisive (83% against 12% for random ones). The rank they encode is no more readable
  than surface counts (59%), though, so seed 1 probably relies on "how close is x to an end"
  information, which fits its weaker composition score.

**Plato's Key across seeds** (exploratory, `experiments/detector_followups/rotation_transfer.py`,
adapted from Hazy Research's 2026 "Plato's Key"). For each pair of seeds, a rotation plus one
global scale was fitted in closed form on x-site activations from 3000 shared puzzles. The
donor's state from a world where only x moved was mapped into the recipient and swapped in. Layers
were chosen on a validation split and scored on a test split.

| Donor → recipient | Rotated donor code | Recipient's own code | Random rotation | Mismatched fit |
|---|---|---|---|---|
| 0 → 1 | **91%** | 88% | 47% | 33% |
| 1 → 0 | **79%** | 80% | 15% | 56% |
| 0 → 2 | **74%** | 57% | 25% | 30% |
| 1 → 2 | **72%** | 57% | 55% | 37% |
| 2 → 0 | 48% | 80% | 21% | 22% |
| 2 → 1 | 67% | 88% | 34% | 45% |

* **The codes are largely interchangeable.** Between seeds 0 and 1, the rotated code works about as
  well as the recipient's own state.
* **Rotations chain.** A→B→C without running B keeps 98–102% of the direct A→C effect in all six
  chains.
* **A better donor can help a weaker recipient.** Seed 0's rotated code moves seed 2 more reliably
  than seed 2's own state does, echoing the paper's "dark memories".
* **Read against the controls, not against zero.** Answers are yes/no, so a patch that merely
  scrambles the network can still look like up to about 50% transfer.

**The Phantasm Slots networks never learned the task, in any of the four slot designs.**

| Slot design | Steps trained | Result |
|---|---|---|
| 8 free slots after the premises | 3000 | chance |
| 6 slots anchored after the premise slots | 8000, then a separate 20,000 | chance |
| Slot Attention (6 competing slots) | 3000 | chance |
| Slot Attention plus raw letter embeddings | 3000 | chance |

This is not a disconnection. Gradients reach the encoder, and changing only the premises changes
the output: by 0.32 at initialisation with Slot Attention, more than the plain network's 0.26.

**Control** (`experiments/detector_followups/bottleneck_control.py`). The same encoder and decoder,
but the decoder may attend to *every* encoder position instead of a few slot vectors. On the same
2–3 letter worlds it breaks through at about the same point as the plain transformer: loss 0.693
until step 1250, 0.677 at 1500, 0.32 at 1750 and 0.21 at 2000. The machinery works. What blocks
learning is the bottleneck itself: having to compress the premises into a few vectors before the
question is known.

**A closing side door breaks the deadlock** (`side_door_config()`; log in
`experiments/detector_followups/side_door_training.out.txt`). The Slot Attention network was given
a gated route from the question straight to the encoded premises. It was fully open for the first
40% of training, closed linearly until 80%, then shut. Accuracy with the door shut is the
bottleneck's own accuracy.

| Steps | Door opening | Accuracy, door at that opening | Accuracy, door shut |
|---|---|---|---|
| 4,500 | 1.00 | 85% | 49% |
| 7,000 | 0.54 | 96% | 51% |
| 9,500 | 0.02 | 96% | 50% |
| 10,000 | shut | 50% | 50% |
| 11,000 | shut | 60% | 60% |
| 12,000 | shut | 77% | 77% |

Nothing moved into the slots while the door was closing: the network kept relying on it and
collapsed to chance once it shut. But then, through the slots alone, it began to learn, going from
50% to 77% in 2,000 steps, where every bottleneck network trained from scratch stayed at chance (one
of them for 20,000 steps). The door did not transfer knowledge. It broke the chicken-and-egg
problem: while it was open, the decoder learned to read premise features, and once a reader
existed, the slots could be shaped to feed it. Continuing for 6,000 more steps with the door shut
(`continue_side_door_shut.py`) took it to 95% held-out accuracy: the first slot network to learn the
task.

**What the detector finds in it** (`results/side_door/`; slot probes in `slot_contents.py`):

| | Plain transformer (seed 0) | Bottleneck network |
|---|---|---|
| 5-letter worlds / composition / stated pairs | 98.9% / 90.5% / 99.6% | 89.1% / 79.5% / 81% |
| Transplant the scene site (before the question) | 1%, does nothing | **90%**, at the 89% ceiling its own accuracy sets |
| Rank readable at the scene site (surface counts: 60%) | 37–41% | 57–58% |
| Right neighbour of each letter readable at the slots | n/a | 41% (untrained copy: 30%) |
| Where the decision forms | x/y sites, layer 2 | x/y sites, layer 1 |
| Transplant only x's rank directions | 2% | 18% (random: 18%) |

Under the fixed rule, composition at 79.5% makes the verdict "no imagination". The internals are
more informative:

* **The slots are a complete standing store.** Transplanting them carries the answer as far as the
  network's own accuracy allows.
* **It is not a readable scene.** Letter positions are no more linearly readable there than surface
  counts make them.
* **It is not a readable copy of the premises either.** Right neighbours read at only 41%.
* **The comparison still happens at the question**, at the asked-about letters (layer 1), just as in
  the plain transformer.

Forcing a standing store produced a standing *memory* in a compressed code, not a standing *model*
of the line.

## 7. What it means

**In the learned mechanism, there is no standing mental model of the scene.** Training did not
produce a picture of the line held ready for any question. It produced *on-demand composition*.
When the question names x, the network assembles at x's position a distributed code for where x
stands relative to the other letters, and computes the answer from it. That code is causally used
and partly structured, but it is not a coordinate, not a per-letter list, and not linearly
decodable well enough to pass the detector's fixed bar. So the verdict stays "something else".

**Across training runs the on-demand code is shared.** Three independently trained networks
all decide at the asked-about letters, none forms a standing scene, and a single rotation lets
one network use another's on-demand code, with chains of rotations composing. That is functional
evidence, in miniature, for the Platonic Representation Hypothesis: what each run invents is
largely a rotated version of the same thing. How well that code supports composition varies from
run to run (73–90%).

**Access × question type** (pre-registered in [INTERPRETATION.md](INTERPRETATION.md); results in
`results/access_x_demand/`). Both networks were trained on a mix of left-of and "is x at position
k?" questions, and learned both (5-letter worlds: plain 92.5%, bottleneck 85%).

* **Plain transformer:** still no standing scene (scene-site transplant 4%, at baseline). Its code at
  x did not become a coordinate (rank 58–68% vs surface 59%; rank directions not causal).
* **Bottleneck:** the slots carry the answer (87%), but rank reads linearly at only 51%, so the
  pre-registered prediction that the slots become a linearly readable scene **failed**.
* **Exploratory:** an MLP probe reads rank from the slots at 0.83. That compares with 0.66 for the
  left-of-only bottleneck, 0.34 for an untrained copy and 0.35 for the raw premises, and against a
  0.60 ceiling for surface counts. So cutting the premises off and asking global questions does
  produce a substantial standing scene, in a non-linear code. Its causal role is not yet isolated.

**The architecture arm points the same way.** Trained from scratch, four designs that force
everything through a slot bottleneck could not learn the task, while the same encoder and decoder
without the bottleneck learned at the plain transformer's pace. A side door that was open during
early training and then shut broke that deadlock (95% held-out accuracy). Even then, the slots did
not become a scene. They hold a compressed, sufficient memory of the premises, and the composing
still happens when the question arrives. On this task, the architecture can force imagination's
*storage*, but not its *form*. It is still a small-scale finding: CPU training, one seed per design,
a few thousand steps.

**What the instrument showed it can do.** It correctly separated a real scene, a surface cheat
whose "scene" was only a copy of the premises, and a network that had learned a positional
shortcut. It located where and when a working network makes its decision.

**Against the introspective self-test (section 10).** Claude reported building the whole line
before reading the question, which would be a standing scene. The small plain transformer does the
opposite. Neither observation says anything about the other: one is an unverified self-report from
a large model, the other is a measurement of a tiny one.

For an attempt to explain these results, with predictions that would test the explanation, see
[INTERPRETATION.md](INTERPRETATION.md).

## 8. Limits

* **Linear probes miss non-linear codes.** "Something else" means no *linearly decodable* scene
  was found that the answer depends on. A non-linear scene would land here too. Subspace
  interchange with a learned rotation (full DAS) would be the next step.
* **One position per site.** A scene spread across many positions would be missed by single-site
  transplants. That is why the scene site is tested at the end of the premises, the only position
  in a causal model that has seen all of them, and why X and Y are tested too.
* **A toy world.** One kind of state (a linear order), small networks, three seeds for the plain
  transformer and one for each slots layout. The instrument
  scales; these particular findings may not.
* **Training budget.** Small CPU-trained networks may solve the task differently than large models
  would, or not at all on longer worlds.

## 9. Applying it to a large language model

The battery carries over to any model whose activations can be read and written, such as an
open-weights model through a hooking library:

1. Choose a task with a crisp latent state and counterfactuals that change one variable at a
   time: spatial layouts, family trees, board positions, a process state.
2. Define the sites. For decoder models: the last premise token (scene), the tokens of the
   asked-about entities, and the answer token. Do this at every layer.
3. Probe for the state at every site, against an untrained or weight-shuffled copy and a surface
   baseline.
4. Patch whole sites from counterfactual prompts (causal tracing), then patch only the probe
   subspace (DAS-style), against random subspaces.
5. Calibrate first. Prompt the same model with a version of the task that is known to be solvable
   by lookup alone, and check that the detector says "no imagination" there.

For a model reachable only through an API, only criterion 1 can be tested, together with
behavioural signatures: invariance to reshuffling, and consistency across counterfactual
prompts. Those can rule imagination out but cannot establish where it lives. That requires the
internals.

## 10. Testing it on Claude

The assistant that built this repository was asked to test itself: does it visualise when it
reasons? Only part of the battery can be run on it here, because a chat session has no access to
its own activations.

**Behaviour (criterion 1).** `experiments/self_test/self_test.py make` generates 34 puzzles with
the answers hidden. Claude answered them in its own reasoning, with no code
(`experiments/self_test/claude_answers.txt`), and the script scored them:

| | Claude | surface heuristic |
|---|---|---|
| questions needing composition | 8/8 | 50% |
| stated pairs | 8/8 | 100% |
| 7-letter worlds | 6/6 | 85% |
| same answer after reshuffling | 8/8 | 94% |

Thirty-four items is a small sample. Even so, it rules out the surface-statistics explanation.
It does not establish imagery: people with aphantasia also solve such puzzles, using verbal or
abstract-spatial strategies, and the surface heuristic shows how a strong-looking score can come
from something else.

**Introspection.** Claude reported that its reasoning on every puzzle took the same form. It
rewrote each `>` premise as `<`, joined the premises into a single line by attaching pieces at
the ends, wrote that line out (`E < L < Y < P < D`), and read the answer off the positions. It
built the whole line before considering the question, which looks like a "standing scene" in its
reasoning trace. It did not report anything it could identify as a picture. Its working
representation was a line of symbols, closer to the aphantasic strategy than to seeing an image.

**Why the self-report does not settle it.** The report is produced by the same system it
describes, and nothing checks it against the mechanism. Anthropic's interpretability research on
introspection, which injects known concepts into a model's activations and asks whether it
notices, has found that Claude models sometimes detect such internal states but unreliably, and
miss them more often than not. A report of "no pictures" is therefore weak evidence in either
direction, much as a probe without an intervention is. The question can only be settled the way
this detector settles it for the small networks: probes and transplants on Claude's own
activations. That requires interpretability access that a chat session does not have.

## References

* Belinkov, Y. (2022). Probing classifiers: Promises, shortcomings, and advances. *Computational Linguistics*, 48(1).
* Elazar, Y., et al. (2021). Amnesic probing: Behavioral explanation with amnesic counterfactuals. *TACL*.
* Geiger, A., Lu, H., Icard, T., & Potts, C. (2021). Causal abstractions of neural networks. *NeurIPS*.
* Geiger, A., Wu, Z., Potts, C., Icard, T., & Goodman, N. (2023). Finding alignments between interpretable causal variables and distributed neural representations. arXiv:2303.02536.
* Hewitt, J., & Liang, P. (2019). Designing and interpreting probes with control tasks. *EMNLP*.
* Li, K., et al. (2023). Emergent world representations: Exploring a sequence model trained on a synthetic task. *ICLR*.
* Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and editing factual associations in GPT. *NeurIPS*.
* Nanda, N., Lee, A., & Wattenberg, M. (2023). Emergent linear representations in world models of self-supervised sequence models. *BlackboxNLP*.
* Ravichander, A., Belinkov, Y., & Hovy, E. (2021). Probing the probing paradigm: Does probing accuracy entail task relevance? *EACL*.
* Vig, J., et al. (2020). Investigating gender bias in language models using causal mediation analysis. *NeurIPS*.
