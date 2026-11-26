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
transplanting that site *does* move its answers (94%). A detector that relied on transplants
alone would call it imaginative. The ranks, though, cannot be decoded there beyond what surface
counts give, and its behaviour on composition is near chance (57%). The verdict correctly says
no.

## 5. Subjects

Both networks are trained **only on the yes/no answer**. Nothing about the scene is supervised.
Every training batch mixes worlds of two to six letters. An earlier run that started with tiny
worlds and grew them is examined too, because it learned something instructive (see results).

* **Plain transformer**: 4 layers, width 64, 4 heads, causal attention over the whole sequence.
  Any imagination it has is an invention of its mechanism.
* **Phantasm Slots transformer**: a 3-layer encoder writes the premises into 8 slot vectors, and
  a 4-layer decoder answers the question while able to see *only* the slots. The architecture
  forces everything through a bottleneck, so transplanting the slots must move the answer if the
  network works at all. The open question is whether what flows through the slots is a scene or
  some other code.

## 6. Results

*Filled in from `results/imagination_report.md` after a run; see section 7.*

## 7. What it means

*See section 6.*

## 8. Limits

* **Linear probes miss non-linear codes.** "Something else" means no *linearly decodable* scene
  was found that the answer depends on. A non-linear scene would land here too. Subspace
  interchange with a learned rotation (full DAS) would be the next step.
* **One position per site.** A scene spread across many positions would be missed by single-site
  transplants. That is why the scene site is tested at the end of the premises, the only position
  in a causal model that has seen all of them, and why X and Y are tested too.
* **A toy world.** One kind of state (a linear order), small networks, one seed. The instrument
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
| questions needing composition | 8/8 | about 57% |
| stated pairs | 8/8 | 100% |
| 7-letter worlds | 6/6 | about 85% |
| same answer after reshuffling | 8/8 | about 94% |

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
