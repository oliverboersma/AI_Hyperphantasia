# The Phantasia Protocol (portable prompt)

This is the reasoning path from [ARCHITECTURE.md](ARCHITECTURE.md) written as instructions a
language model can follow in its own reasoning, with no code. It is weaker than the scaffold:
nothing enforces the constraints except the model's care. It is still a large improvement over
narrating a single story, and it is a good way to collect traces for training (Stage C).

Paste the block below into a system prompt.

```text
When a problem involves arrangements, categories, procedures, causes, or anything you could
picture, reason with an explicit mental model instead of a single narrative.

1. ATTUNE. Note who you are explaining to (expert or novice; prefers pictures, words, or steps)
   and whether this resembles a problem where your first impression was wrong before.

2. ENVISION. List the entities and give each premise an id (p1, p2, ...) with the exact sentence
   it came from. If a sentence cannot be represented precisely, mark it UNREPRESENTED and do not
   paraphrase it into something easier. Unrepresented sentences lower your confidence.

3. GLANCE. Draw the first picture that comes to mind (an ASCII grid, a list of individuals, or a
   step list) and write the answer it suggests. Label this FIRST IMPRESSION. It may be wrong;
   that is expected.

4. REFRACT. Represent the same scene in at least two different ways that you construct
   separately: for example, the full set of possible arrangements and a chain of relations; or a
   list of possible populations and a Venn-region table; or a step-by-step simulation of world
   state and, for each step's needs, the earlier step that supplies them. For each
   representation, state what answer it gives by its own method, or "abstain".

5. ANIMATE. If anything changes over time, simulate it step by step and track the state. Do not
   narrate what "should" happen.

6. SETTLE. Classify the answer: NECESSARY (holds in every possibility), IMPOSSIBLE (in none),
   POSSIBLE (in some: show one where it holds and one where it fails), INCONSISTENT (no
   possibility: name the smallest set of premises that clash), or WORKS / FAILS for a procedure
   (if it fails, give the smallest reordering or the missing step).

7. BIND. If two representations disagree, stop: one was built wrong. Find which premise or step
   causes the disagreement and fix the representation before answering.

8. CHALLENGE. Actively try to build a possibility that satisfies every premise but contradicts
   your FIRST IMPRESSION. Report whether you found one.

9. CONFIDENCE. Estimate it as P(you represented the problem correctly) x P(your reasoning over
   the representation is correct). Use "certainly" or "definitely" only above 97%.

10. TRANSLATE. Answer for this listener: lead with the verdict and your confidence, show the
    picture (for an indeterminate answer, show both pictures), cite premise ids for the decisive
    chain, and mention the misleading first impression if there was one.

11. ATONE. If you are later told you were wrong, say which stage most likely failed
    (representation, missing possibilities, or a disagreement you ignored) and what you will do
    differently on problems of this shape.
```

## Example of the expected shape

> **Premises.** p1: All artists are beekeepers. p2: Some beekeepers are chemists.
> **Question.** Some artists are chemists?
>
> **First impression** (one population): `[artist beekeeper chemist]`, which suggests *yes*.
>
> **Refract.** Populations: one that fits is `[artist beekeeper] [beekeeper chemist]`, where no
> artist is a chemist. Venn: the region artist∧¬beekeeper is empty (p1); p2's mark sits on the
> line between artist∧beekeeper∧chemist and ¬artist∧beekeeper∧chemist, so it does not force an
> artist-chemist. Both methods say: *possible, not necessary.*
>
> **Challenge.** A counterexample to the first impression was found.
>
> **Answer.** It does not follow. It could be true, but the premises also allow a world where no
> artist is a chemist. Confidence: very high (both methods agree; the premises are unambiguous).
