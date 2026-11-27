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

The evidence about seed 0's code at x's site, at layer 2:

* It is causally decisive: transplanting it moves the answer 81% of the time.
* It is not a coordinate: transplanting only the rank directions does nothing (2%).
* Most of its effect lies in the span of "is letter L left of x?" directions (78%, against 38% for
  a random span of the same size).
* No single letter's direction is specific: editing y's own direction (18%) does no better than
  editing an uninvolved letter's (17%).

A consistent reading is that x's site holds a superposed set of x's neighbourhood: the letters
reachable to its left and to its right within a couple of hops, each carried in its letter's
embedding direction with a sign. y's site holds y's neighbourhood the same way, and the answer is
computed from the overlap of the two. That explains the lack of specificity. Moving x changes the
membership of several letters at once, so flipping only y's entry rarely changes the overlap
enough to change the answer.

The other observations fit this reading:

* **Seed differences.** Seed 1 appears to lean on a cruder feature, nearness to an end, which is
  causal through a few rank-like directions but weak on middle letters. That fits its lower
  composition score (83%).
* **Shared codes.** The rotation transfer says that, whatever their probe-visible differences, the
  seeds' codes mostly share one geometry. That is what you would expect if the code is built from
  the letter embeddings themselves: the neighbourhood set is the natural object for this computation,
  so independent runs converge on it, which is a small-scale version of the Platonic Representation
  Hypothesis.

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
4. **Language models compose on demand without a scratchpad.** With a single question after a static
   description, an LLM should decide at the question tokens. With chain-of-thought, the written chain
   should act as the standing scene, and editing it should change the answer.

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
