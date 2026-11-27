# Making sense of the results

This is an attempt to explain *why* the experiments came out as they did
([IMAGINATION_DETECTOR.md](IMAGINATION_DETECTOR.md)), and what that implies. It is reasoning, not
new data: each claim below says what supports it and what would refute it.

## The puzzle

Three things need explaining together:

1. **No standing scene.** Every plain transformer that learned the task built no model of the
   whole scene before the question. Each composed an answer at the asked-about letters when the
   question arrived.
2. **Forcing one blocks learning.** Four designs that force a scene through slots before the
   question never learned. The same parts without the bottleneck learned at normal speed.
3. **Shared on-demand codes.** Independently trained networks invent on-demand codes that are
   largely rotations of one another.

## The central idea: a standing scene is an amortisation

A standing mental model is a cache. Building it costs work up front: here, placing every letter,
which means composing the whole chain. It pays off only if that work is reused, by many
questions, by actions, by simulation, or after the original information is gone.

In our task none of that holds:

* **One question.** Each puzzle asks one yes/no question about two letters (four during training).
* **The premises stay visible.** The question tokens can attend to every premise directly.
* **Only part of the scene is needed.** One question needs the relation between two letters, not
  the whole order.

So the cheapest sufficient computation is to compose on demand: when the question names x, look up
x's premises and their neighbours, and compare with y. Gradient descent finds the cheapest
sufficient computation, so that is what all three networks did.

This reframes the original question. Whether a system "has imagination" may be less a fixed
property of its architecture than a strategy it adopts when imagination pays: when a scene is
queried densely and unpredictably, manipulated, or must outlive its source. That is a familiar
picture of human imagery too. People with aphantasia solve these puzzles well by working out
relations as needed, while vivid imagery earns its keep in memorisation (the method of loci),
rehearsal and mental simulation, all of which reuse one scene many times.

## Why on-demand is also easier to *learn*

Cost explains the preference, but not why the bottleneck networks failed completely. That needs
a learnability argument, and the evidence narrows it down.

* **On demand, the search starts with a foothold.** At initialisation the question's letter x and
  the premise letter x share an embedding, so the attention that matches them gets a usable
  gradient immediately. Once x's premise is found, its role (left or right of the sign) can be read
  off its position. The first useful circuit is short, and its pieces get signal straight away.
* **A standing scene in a bottleneck is a chicken-and-egg problem.** The encoder must write a code
  into a few vectors before the question exists, and the decoder must learn to read that code. At
  the start neither has anything to work with: a random code carries nothing the decoder can use,
  and a random decoder gives the encoder no signal about what to write.
* **The hard part is binding under compression.** Squeezing "a is left of b" into a slot shared with
  other tokens needs a role–filler binding scheme, which slot averaging does not provide for free.
  This matches how the controls came out. Restoring letter identity alone (raw embeddings in the
  slots) did not help, while removing the compression (letting the decoder see every token) restored
  learning at once. With one vector per token, binding comes free from position; with mixtures,
  it has to be invented.

## What the on-demand code probably is

*Revised after the alignment search at the end of this document. The first reading here, a
superposed set of x's neighbours, did not survive it.*

The evidence about seed 0's code at x's site, at layer 2, from the first round:

* It is causally decisive: transplanting it moves the answer 81% of the time.
* The linear rank probe's directions are not causal: transplanting only those does nothing (2%).
* Most of its effect lies in the span of "is letter L left of x?" directions (78%, against 38% for
  a random span of the same size).
* No single letter's direction is specific: editing y's own direction (18%) does no better than
  editing an uninvolved letter's (17%).

The first reading was that x's site holds a set of x's neighbours (the letters to its left and
right, each in its own embedding direction), y's site holds y's, and the answer reads the overlap.
That reading predicts that transplanting x's code from a world with a different order carries
that world's relations, so the answer becomes that world's own answer. It does so only 41% of
the time.

**Current reading: a coordinate, built on demand.** Alignment search finds that one direction of the
64 carries almost all of the site's effect. It holds x's place on the line as a number, sharp at the
two ends and crowded in the middle, and the network compares it with y's place. The linear probe's
rank directions missed it. The 26 membership directions caught most of it because a
direction that orders the letters correlates with every "is L left of x?" relation.

Two older observations read differently now:

* **Seed differences.** Seed 1's code looked like nearness to an end. The alignment search on all
  three seeds confirms it and makes it precise. Every seed holds a coordinate. They differ in how
  much of the line its main direction resolves: one end (seed 2), both ends (seed 1), or both ends
  and a graded middle (seed 0).
* **Shared codes.** The rotation transfer between seeds fits a shared coordinate well: a single
  ordered axis is the natural object for this computation, so independent runs converge on it.

## How this fits other work

* **Othello-GPT** forms a standing, linearly decodable and causally used board state. That fits the
  amortisation view rather than contradicting it. In Othello every move is a question about the
  whole board, and the queries are dense and unpredictable, so maintaining the state is the cheapest
  solution. Our puzzles ask one sparse question at the end.
* **Plato's Key** swaps the key built at a subject token, which is also an on-demand construct at
  the point of use, and finds it transfers by rotation. Our transfer result is the same phenomenon in
  a reasoning task.
* **Diagrams and chain-of-thought.** Larkin and Simon's argument for diagrams is that they are
  external, standing representations, cheap to re-read. For a language model, chain-of-thought is
  exactly such a medium: once a line like `E < L < Y < P < D` is written, every later token can read
  it. That is also what Claude reported doing in the self-test. If this view is right, a language
  model's standing scene lives most naturally in the text it writes, not in a single internal state.

## What it means for the hyperphantasia architecture

The scaffold in [ARCHITECTURE.md](ARCHITECTURE.md) already externalises the scene (the possibility
space, the pictures, the proof graph) instead of hoping one forms inside the network. These results
support that choice: internal standing scenes are expensive to build and hard to learn, while
external ones are cheap and easy to check. They also suggest a refinement, *imagination on
demand*: build only the part of the scene that the current question needs, and pay for a full
standing scene only when it will be reused, for many questions, simulation, memorisation or
explanation.

## Predictions

Each of these would test the account, and most are cheap to run here:

1. **Dense questions produce a standing scene.** Interleave questions with the premises, so the state
   must be ready at every point (as in Othello). A scene should then appear at the premise positions,
   decodable and causally used.
2. **Removing access forces a scene, but only if it is learnable.** If the premises leave the
   question's view (a shrinking attention window, or the closing side door now training), a standing
   scene must form. Whether one does depends on giving the reader a foothold first. The side-door run
   tests this: if the decoder learns to read token-level features while the door is open, closing it
   should pull those features into the slots, and accuracy with the door shut should rise as it
   closes.
3. **The answer reads an overlap.** Transplanting the neighbourhood span at both x's and y's sites
   together should move the answer far more reliably than either alone, and close to always.
   (Derived from the neighbourhood reading, which the alignment search later disfavoured.)
4. **Language models compose on demand without a scratchpad.** With a single question after a static
   description, an LLM should decide at the question tokens. With chain-of-thought, the written chain
   should act as the standing scene, and editing it should change the answer.

## Pre-registered test: access × question type

This section was written before the runs it describes.

**A refinement of the account.** In a transformer, recomputing per question costs no extra time:
every position has the full depth, and the question tokens can always re-read the premises. So
"many questions" alone creates no pressure for a standing scene. What should matter is:

* **Access:** can the question still see the premises?
* **Demand:** does the question need a *global* fact (x's place in the whole line) or only a local
  relation (x versus y)?

The bottleneck network showed that removing access with only local questions produces a compressed
memory, not a scene.

**Design.** Train two networks on a mix of questions. Half are "is x left of y?"; half are "is x at
position k?", which needs x's place in the whole line. Evaluate with the usual detector on
left-of questions.

| | Local questions only (done) | Mixed with position questions (new) |
|---|---|---|
| Question sees the premises (plain transformer) | on-demand, not a coordinate | Prediction A: still no standing scene before the question; the code at x's site becomes coordinate-like (rank readable well above surface counts, rank directions causal) |
| Premises only through slots (side-door bottleneck) | compressed memory, not a scene | Prediction B: **the slots become a scene**: rank readable from the slots well above surface counts (above 0.8), and transplanting the slots carries the answer |

**How the result would bear on the account.**

* **B holds:** the account survives, refined. A scene forms when access is removed *and* the
  questions need global facts.
* **B fails** (the slots stay a non-scene memory even when position questions must be answered from
  them): the account is wrong about what makes a scene pay, at least for networks this size.
* **A fails** (a standing scene appears in the plain transformer): access does not matter the way
  the account says.

## Results of the pre-registered test

Both networks learned the position questions (5-letter worlds: plain 92.5%, bottleneck 85%), so
the manipulation worked (`results/access_x_demand/`).

| Prediction | Pre-registered criterion | Result |
|---|---|---|
| A: plain network, still no standing scene | scene-site transplant at baseline | **Held**: 4%, at baseline |
| A: plain network, code at x becomes coordinate-like | rank readable well above surface counts, rank directions causal | **Failed**: 58–68% (surface counts 59%); rank directions 6% vs 9% random |
| B: bottleneck slots become a scene | rank linearly readable from the slots above 0.8 | **Failed**: 51% (surface counts 60%), while transplanting the slots carries the answer (87%) |

Also, at x's own token, rank is no more readable when the question asks for x's position than when
it asks whether x is left of y (plain 71% in both; bottleneck 45% in both).

**Exploratory follow-up (not pre-registered).** A linear probe can miss a non-linear code, so a
small MLP probe was run (`experiments/detector_followups/mlp_scene_probe.py`). Because an MLP can
compute ranks itself, the baseline is the same probe on a raw, lossless encoding of the premises.

| Input to the MLP probe | Rank accuracy |
|---|---|
| Bottleneck slots, mixed questions | **0.83** |
| Bottleneck slots, left-of questions only | 0.66 |
| Untrained copy's slots | 0.34 |
| Raw premises (lossless one-hot) | 0.35 |

What the follow-up adds:

* **The probe is not doing the work.** On the raw premises it reaches only 0.35.
* **The slots hold more than surface counts.** Counts identify only the two ends of the line, which
  caps any probe on them at 0.60.
* **Global questions make the store more scene-like.** The bottleneck slots do hold a scene of
  positions in a non-linear code, and adding position questions raised it from 0.66 to 0.83.

**Reading.** As pre-registered, B failed, and the account as stated (a linearly readable scene forms
when access is removed and questions need global facts) is wrong. The exploratory result rescues its
idea in a different form. Removing access and adding global questions does produce a substantial
standing scene, encoded non-linearly. It is not yet shown to be the part of the slots the answer is
computed from (the slots as a whole are, at 87%). With access retained (the plain network), no
standing scene forms at all, whatever the questions. The next test is causal: find the non-linear
scene's directions (a learned rotation, or interventions guided by the MLP probe) and check that
editing them moves the answer.

## Causal test of the non-linear scene

`experiments/detector_followups/scene_edit.py`. The slots were edited by gradient steps on the MLP
rank probe until it read a counterfactual scene (x swapped with a letter beyond y, all other letters
kept), and the network then ran on the edited slots. Results for the bottleneck trained on mixed
questions, on 578 puzzles it answers correctly in both worlds:

| Edit to the slots | Answer follows the edit | Unmoved pair keeps its answer | "x at new position?" → yes |
|---|---|---|---|
| none | 0% | 100% | 4% |
| real slots of the counterfactual world (ceiling) | 100% | 72% | 94% |
| **guided by the real probe** | **22%** | 80% | **16%** |
| guided by a shuffled-label probe | 0% | 100% | 3% |
| random edit of the same size | 7% | 85% | 6% |

The edit worked on its own terms: afterwards the probe read the counterfactual ranks from the
slots 99% of the time, for a change of 14% of the slots' size.

* **The network partly follows the edited scene.** Probe-guided edits flip the answer about three
  times as often as random edits of the same size (22% vs 7%), and make "is x at its new position?"
  come out yes 16% of the time, against 6%. With 578 puzzles these gaps are far beyond chance.
* **Only partly.** The real slots of that world flip the answer every time.
* **Not selectively.** The edit disturbs answers about letters it did not move slightly more than a
  random edit does (80% vs 85% unchanged).
* **The left-of-only bottleneck shows the same pattern, a little weaker** (probe 0.67; flips 17%
  vs 8%).

**Reading.** The non-linear scene is partly the thing the network computes from: editing it moves
the answers the right way, but most of what drives them lies outside the directions the probe reads.
The forced store carries a scene of positions, partially used, mixed with other information the
answer also depends on. It is not a clean picture that the answer is read from.

## Where the answer is actually read from: alignment search

The probe-guided edit trusts the probe's directions. Distributed alignment search instead asks the
network which directions it uses. `das_slots.py` learns an orthonormal subspace of the flattened
slots (6 × 64 = 384 dimensions) such that transplanting only that subspace from a source world (the
same letters in a different order) makes the frozen network answer as the source world would.
Questions mix left-of and "is x at position k?". Interchange accuracy is measured on 2,000 held-out
pairs whose two answers differ.

| Transplant from the source world | Answer follows the source |
|---|---|
| all 384 dimensions | 0.862 |
| learned 8-dimensional subspace | **0.830** |
| learned 32 / 128 dimensions | 0.831 / 0.862 |
| random 8 / 32 / 128 dimensions | 0.172 / 0.195 / 0.341 |
| **everything except the learned 8** (376 dimensions, `das_complement.py`) | **0.188** |

Where the letter positions can be read (MLP rank probe; raw premises 0.35, all slots 0.83):

| Read from | k = 8 | k = 32 |
|---|---|---|
| the learned causal subspace only | 0.641 | 0.736 |
| everything else | 0.831 | 0.830 |

* **The network computes from a tiny code.** Eight directions out of 384 carry 96% of what the
  whole slots carry (0.830 of 0.862). Larger subspaces add almost nothing.
* **The readable scene is mostly idle.** The other 376 dimensions hold a scene the probe reads as
  well as from the full slots (0.83), yet swapping them alone barely moves the answer (0.188, about
  the level of 8 random dimensions and below 128 random ones). Base and source share their letters,
  so what differs between them is the order, and that order information outside the 8 directions
  goes unread.
* **The causal code is an ordering code, though not the most readable one.** Positions read from
  the 8 dimensions at 0.64, well above the raw premises (0.35), so the used code does carry the
  order, in a form a small probe decodes less easily than the network's own decoder does.
* **It is spread across slots.** Every slot holds part of the subspace (0.8 to 2.0 of its 8
  dimensions), and each learned direction has only about a third of its weight in its biggest slot
  (`das_structure.py`). It is not one letter per slot.

**Reading.** This explains the 22% from the probe-guided edit. The probe's directions lie mostly in
the large, readable, idle part of the slots and only leak into the 8 dimensions the answer depends
on. The forced store holds the order twice: once as a compact code the decoder uses, and once as a
redundant, easily readable copy it ignores. So a probe that finds a scene in a network does not show
that the network uses it. Here the scene a probe found was real but mostly a by-product, and the
scene the network used was much smaller than the one on display. One network and one seed; the
redundant copy might be a leftover of training through the side door, which the same search on
other runs could check.

## The on-demand code, read by alignment search

`das_plain.py` and `das_plain_coordinate.py`, on the plain transformer (seed 0), at x's site in
layer 2 (64 dimensions). It is the same search as for the slots, on the code the plain network
builds when the question names x.

**Size.** "Move only x" pairs (x swaps with a letter beyond y, so y keeps its place), on held-out
pairs whose answers differ:

| Transplant | Answer follows the other world | Random directions of the same size |
|---|---|---|
| the whole site | 0.816 | |
| 1 learned direction | **0.802** | 0.017 |
| 2 learned directions | 0.846 | 0.017 |
| 4 / 8 / 16 learned directions | 0.848 / 0.850 / 0.851 | 0.034 / 0.036 / 0.102 |

**Coordinate or relations?** The other world now has the same letters in a new order. A transplant
of x's code then predicts two different answers. If the code is a **coordinate**, the network
compares x's place in the other world with y's place in its own. If the code holds
x's **relations**, it gives the other world's own answer. Each hypothesis is scored on the pairs
where its answer differs from the base answer:

| Transplant | follows "relations" | follows "coordinate" |
|---|---|---|
| the whole site | 0.41 | **0.73** |
| 1 learned direction | 0.41 | **0.74** |
| 8 learned directions | 0.46 | **0.81** |
| 8 directions fitted to each hypothesis | 0.42 | **0.76** |

A first version of this comparison scored the pairs where the two hypotheses disagree. That was
confounded, because there the unpatched network already agrees with "coordinate" 85% of the time.

**The direction itself.** Its mean value for x at ranks 0 to 4 is +17.6, +4.5, +0.5, −4.4 and
−19.2 (standard deviations 3 to 5). Its rank correlation with x's rank is −0.93, against 0.79 for
the best linear guess from surface counts, and it keeps a correlation of −0.57 with rank once
surface counts are partialled out.

**Placing x by setting the number.** x's code is set to its mean value for rank r, everything
else is left alone, and the test counts how often the answer changes as if x sat at r. Only cases
where that must change the answer are counted:

| Directions set | r = 0 | 1 | 2 | 3 | 4 | overall |
|---|---|---|---|---|---|---|
| 1 | 0.95 | 0.57 | 0.12 | 0.59 | 0.97 | 0.75 |
| 2 | 0.98 | 0.59 | 0.46 | 0.57 | 0.98 | 0.80 |
| 4 | 0.98 | 0.60 | 0.47 | 0.58 | 0.98 | 0.81 |
| 1 random direction | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.002 |

* **The code is almost one number.** One direction out of 64 carries 98% of what the whole site
  carries. Two carry slightly more than the whole site, which also holds information that works
  against the transplant.
* **It is a coordinate.** A transplant carries x's place from the other world, and the network
  compares it with y's place in its own world (0.73 to 0.81). It does not carry the other world's
  relations (0.41 to 0.46).
* **It is composed, not counted.** The values come in rank order and track rank much better than
  surface counts do. They stay related to rank once those counts are removed.
* **It is a number line, sharp at the ends and blurry in the middle.** The ends lie far apart and
  the middle three ranks are crowded. Setting the number puts x at either end almost every time
  (95 to 98%), but into the middle much less often (12% with one direction, 46% with two). A
  second direction carries much of the middle, and more directions add nothing.

**Reading.** Asked about x, the plain transformer imagines one thing: where x sits on the line, as
a number. It then compares that with where y sits. It never builds the line, only the point it
was asked about. That is the smallest scene that answers the question, built when the question
arrives, which is what the amortisation account says an on-demand system should do.

This revises the first round's "not a coordinate". The linear rank probe's directions were simply
not the ones the network uses. Together with the slots, where the readable scene was the one not
used, probes and causal structure come apart in both directions here. It also bears on prediction A
of the pre-registered test: the left-of-only network already held a coordinate at x's site, which
the linear tools could not see. That puts the detector's fixed verdict, "something else", in
context, though the verdict stands as registered.

### Three seeds

The same two scripts on seeds 1 and 2 (`das_plain_seed{1,2}.out.txt`,
`das_plain_coordinate_seed{1,2}.out.txt`), at the same site:

| | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| Composition questions (from the battery) | 90.5% | 83% | 73% |
| Move only x: whole site / 1 learned direction / 2 | 0.82 / 0.80 / 0.85 | 0.88 / 0.89 / 0.92 | 0.62 / 0.60 / 0.64 |
| Reorder, whole site: relations / coordinate | 0.41 / **0.73** | 0.51 / **0.80** | 0.33 / **0.56** |
| Reorder, 8 learned directions: relations / coordinate | 0.46 / **0.81** | 0.53 / **0.91** | 0.33 / **0.60** |
| Main direction, mean value at ranks 0 to 4 | +17.6, +4.5, +0.5, −4.4, −19.2 | −16.9, +0.2, +0.3, +0.4, +19.9 | −3.6, −1.9, −1.7, −1.6, +10.6 |
| Its correlation with rank beyond surface counts | **−0.57** | 0.02 | 0.02 |
| Placement with 1 direction, r = 0 / 2 / 4 (overall) | 0.95 / 0.12 / 0.97 (0.75) | 1.00 / 0.70 / 1.00 (0.83) | 0.41 / 0.22 / 0.91 (0.49) |
| Placement with 2 directions, r = 2 (overall) | 0.46 (0.80) | 0.86 (0.90) | 0.29 (0.54) |

* **Robust: a low-dimensional coordinate.** In every seed, one or two directions carry the whole
  site's effect, and a transplant carries x's place rather than the other world's relations.
* **Not robust: how much of the line the main direction resolves.** Seed 2's marks only the last
  letter. Seed 1's marks both ends and is flat in the middle. Seed 0's marks both ends and grades
  the middle.
* **The ends need no composing.** In every puzzle, the first letter is the only one never stated to
  be right of anything, and the last the only one never stated to be left of anything
  (`ends_from_counts.py`). So an end detector is a surface count. That is why seeds 1 and 2's main
  directions carry nothing beyond surface counts (0.02), while seed 0's graded middle does (−0.57).
* **The ranking matches composition.** One end, both ends, then both ends and a graded middle: in
  the same order as the three seeds' composition scores (73%, 83%, 90.5%). The match is not exact
  once a second direction is included. Seed 1's second direction places middle letters better than
  seed 0's (86% vs 46%), although seed 1 composes less well overall.

**Reading.** All three networks answer by turning the asked-about letter into a position on a line,
built when the question names it. Training decides how much of that line gets built. The cheap
part, which end a letter is at, can be read straight off the premises, and every seed builds at
least some of it. The middle needs real composition, and the seeds differ in how much of it they
put into the coordinate. They also differ in how well they compose overall.

**Limits.** Setting a mean value is a blunt intervention. Seed 0 answers middle comparisons almost
perfectly, though its coordinate is blurry there, so it must get finer middle information from
somewhere else. That may come straight from the stated premises at y's site.

## What could make this wrong

* **Hyperparameters.** The learning rate, initialisation and batch size were tuned for the plain
  transformer. "Hard to learn" might only mean "hard with these settings".
* **Sample size.** There is one seed per bottleneck design, and three for the plain transformer.
* **Weak supervision.** One answer bit per question. Richer feedback might make a standing scene
  easier to learn.
* **Measurement limits.** Linear probes and single-position transplants can miss non-linear or
  spread-out codes, so "no standing scene" means none that these tools can see. The tools did see
  the scene engine's scene, and the scene site after the premises is the only position in a causal
  model where a complete scene could live before the question.
* **The amortisation account is an interpretation.** Prediction 1 is the cleanest test. If dense
  questions still produce no standing scene, the account is wrong.
