# AI Hyperphantasia

A reasoning path for AI models that **build mental models, view them through several "senses"
at once, and feel when something doesn't fit**. It is modelled on how people who think in vivid
imagery reason and memorise: hold a model in mind, walk around it, run it, check it against
itself, and learn from where the first impression was wrong.

* **Design:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) covers the idea, the reasoning loop,
  the signal bus, self-attunement, communication, the memory palace, the path to a native
  architecture ("Phantasm Slots"), evaluation, and limits.
* **Portable prompt:** [docs/PROTOCOL.md](docs/PROTOCOL.md) is the same path as instructions any
  language model can follow.
* **Working prototype:** the `hyperphantasia` package runs the full loop. It has no dependencies,
  and Claude is optional for open-ended language.

## The reasoning path

```
ATTUNE → ENVISION → GLANCE → REFRACT → ANIMATE → SETTLE → BIND → CHALLENGE → TRANSLATE → ATONE
```

| Stage | What happens |
|---|---|
| **Attune** | Recall the listener, my track record in this domain, lessons, and any problem of the same shape |
| **Envision** | Build a scene in which every entity and premise has a stable id and a source sentence; unreadable sentences become holes, not guesses |
| **Glance** | Form the first picture and read a tentative answer off it (System 1, kept on purpose) |
| **Refract** | Render the scene through independent channels: picture, possibility space, proof graph, Venn regions, simulation, causal links, timeline, memory palace |
| **Animate** | Run any dynamics forward |
| **Settle** | Decide over *all* possibilities: necessary / impossible / possible (with witness and counterexample) / inconsistent (with the minimal clash) / works / fails (with a repair) |
| **Bind** | Cross-check channels: agreement is corroboration, disagreement is dissonance (an internal fault) |
| **Challenge** | Confront the first impression with the counterexample search |
| **Translate** | Explain as a guided tour for this listener, at the confidence actually earned (an honesty check rejects overclaiming) |
| **Atone** | Store lessons from my own surprises, file the shape in the memory palace; on feedback, calibrate and blame the right stage |

## Quick start

```bash
pip install -e .
```

```bash
python -m hyperphantasia "The cup is left of the plate. The fork is left of the plate. Is the fork left of the cup?"
```

```
Not determined. It is possible that fork is left of cup, but not necessary.

I'm highly confident (confidence 95%).

Careful: the first picture that comes to mind makes it look settled. That picture fits the
premises, but it is not the only one.

An arrangement that fits every premise, where it holds:
    +-------+-------+-------+
    | fork~ |  cup~ | plate |
    +-------+-------+-------+
    ~ = blurry: position not fixed by the premises

Another that fits every premise, where it does not:
    +-------+-------+-------+
    |  cup~ | fork~ | plate |
    +-------+-------+-------+
```

Add `--trace` to see every stage of the reasoning path.

More examples:

```bash
python -m hyperphantasia "All artists are beekeepers. Some beekeepers are chemists. Does it follow that some artists are chemists?" --listener "words"
```

```bash
python -m hyperphantasia examples/tyre.txt
```

```bash
python -m hyperphantasia "The lamp is directly above the cup. The cup is left of the plate. The plate is in the same row as the cup. Where is the lamp relative to the plate?"
```

The tyre example is a procedure with a hidden ordering bug. The loop rehearses it, finds that
"Loosen the nuts" needs the car on the ground after "Jack up the car" has lifted it, proposes the
minimal fix, and lays out the corrected procedure as a memory-palace route.

From Python:

```python
from hyperphantasia import PhantasiaLoop, ListenerModel

loop = PhantasiaLoop(state_dir=".phantasia")          # self-model, calibration and palace persist
insight = loop.think("All cats are mammals. All mammals are animals. Are all cats animals?",
                     ListenerModel(prefers="words", expertise="expert"))
print(insight.status, insight.confidence)
print(insight.report())                               # the full reasoning path
loop.feedback(insight, correct=True)                  # extrinsic atonement / calibration
```

### With Claude (open-ended language)

```bash
pip install -e ".[claude]"
```

```bash
python -m hyperphantasia "On my desk the stapler sits to the left of the monitor, and my mug is somewhere left of the monitor too. Is the mug to the left of the stapler?" --claude --intuition --trace
```

Claude (`claude-opus-5-5`) reads the problem into a JSON-schema-constrained scene, gives a fast
verbal first impression as an extra channel, re-checks the scene when that impression disagrees
with the model, and rewrites the explanation for the listener. **The model decides the answer,
not the verbal guess**, and rewrites that overclaim certainty are rejected. Requests opt in to
server-side refusal fallback (`fallbacks: "default"`). This path is unit-tested against a stub
client and has not been run against the live API here.

## Results from the prototype

```bash
python examples/bench.py
```

```
Syllogisms (256 forms)
  first impression correct: 152/256 (59%); it accepts 104 invalid form(s)
  full imagination:         256/256 (100%), 0 invalid forms accepted

Ordering puzzles (400, independent brute-force oracle)
  first impression correct: 134/400 (34%)
  full imagination correct: 400/400 (100%)

Noisy perception: half the scenes have one premise read backwards (feedback after each)
                      accuracy  raw conf  calibrated  Brier raw  Brier calibrated
  first 100             79.0%     97.4%       83.0%     0.2036            0.1726
  last 400              80.5%     97.1%       78.9%     0.1889            0.1576
```

* Single-picture reasoning, the "aphantasic" baseline, cannot represent "undetermined" and
  accepts illusory inferences.
* The engine finds exactly the **24 classically valid syllogisms** (15 without existential
  import), and its two independent channels agree on all 256 forms (`tests/test_categorical.py`).
* When perception is silently corrupted, reasoning stays exact but answers go wrong. Feedback
  teaches the system how far to trust itself, and stated confidence drops from 97% to about the
  real 80%.

```bash
python -m pytest
```

This runs 26 tests and needs no network.

## Does imagination exist in the architecture, the mechanism, or neither?

[docs/IMAGINATION_DETECTOR.md](docs/IMAGINATION_DETECTOR.md) describes an instrument that answers
this for a neural network. It trains small transformers on hidden-order puzzles, using only the
yes/no answers, then asks three questions:

* **Behaviour:** can the network answer questions that need premises combined?
* **Representation:** can each item's position be read out of its activations, beyond what an
  untrained copy or surface counts give?
* **Causation:** if activations are transplanted from a world with the opposite answer, either
  whole or only the decoded "position" directions, does the answer follow?

The detector is calibrated first on a hand-built scene engine (it must say *imagination*) and a
surface-heuristic cheater (it must say *no*).

```bash
pip install -e ".[detect]"
```

```bash
python -m hyperphantasia.detect
```

This writes `results/imagination_report.md`. Results from our run (one seed, CPU):

* **Calibration:** the detector calls the scene engine a standing scene and the surface heuristic
  no imagination.
* **The plain transformer** (98.9% accurate) has **no standing scene**. Nothing before the
  question affects the answer. It builds what it needs **on demand**: when the question names a
  letter, it assembles a code for that letter's place relative to the others and answers from it.
  The detector's fixed tests could not read that code, so the verdict is "something else". A later
  alignment search found what it is: mostly **one number**, x's place on the line. It is sharp at
  the ends and blurry in the middle, and the network compares it with y's place. One learned
  direction carries 80% of the effect, and setting it moves x to either end 95–98% of the time. The
  network imagines only the point it is asked about.
* **Across three training runs**, no network builds a standing scene, and a single rotation lets
  one run's on-demand code stand in for another's (91% transfer between seeds 0 and 1, against
  88% for the recipient's own state), in the spirit of Hazy Research's "Plato's Key". How well
  each run composes premises varies (73–90%).
* **The Phantasm Slots networks**, which force everything through a slot bottleneck, never learned the
  task from scratch, in four slot designs. A side door, open early in training and then shut, let
  one learn (95%). With left-of questions only, its slots hold a compressed memory rather than a
  scene. A **pre-registered test** then added global questions ("is x at position k?"):
  * **Plain transformer:** still forms no standing scene.
  * **Bottleneck:** a linear readout still finds no scene in its slots, so the prediction failed as
    registered. But a non-linear probe reads positions from them at 83% (raw premises: 35%). Cutting
    off access and asking for global facts does create a standing scene, in a non-linear code.
  * **Causal test:** editing that scene flips the network's answer three times as often as a random
    edit of the same size (22% vs 7%), but far less than the full slots do (100%). It is partly
    used, not a clean picture.
  * **Alignment search** explains why. The network reads its answer from just 8 of the 384 slot
    dimensions (83% transfer, against 86% for all of them). The readable scene sits in the other
    376, and swapping those alone barely moves the answer (19%). The store holds the order twice: a
    small code that is used and a large, readable copy that is not.

## Layout

```
hyperphantasia/
  scene.py          binding registry: entities, premises with ids and provenance
  domains/          imaginative media, each with independent channels
    spatial.py      line and grid; possibility space + transitive-closure proofs
    categorical.py  syllogistic populations + Venn shading
    process.py      simulation, causal links and threats, repair, timeline, memory palace
  signals.py        vividness, dissonance, surprise, corroboration, confidence
  attune.py         calibration ledger, self-model, listener model, honesty invariant
  palace.py         memory palace of problem shapes (analogy)
  translate.py      explanation as a guided tour of the model
  loop.py           the Phantasia Loop
  schema.py         validation between perception and the scene
  backends/         offline restricted-English parser; Claude backend
  detect/           imagination detector: test worlds, subjects, nets (JAX), probes, battery, verdict
docs/               architecture, portable protocol, imagination detector
examples/           tyre procedure, benchmark
tests/
```
