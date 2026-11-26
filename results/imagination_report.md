# Imagination detector report

Verdict per subject, from behaviour, linear probes, and interchange interventions. See docs/IMAGINATION_DETECTOR.md for the method.

| subject | verdict |
|---|---|
| known positive: scene engine | standing scene |
| known negative: surface heuristic | no imagination |
| plain transformer | something else |
| plain transformer, packed premises (learned a positional shortcut) | no imagination |

## known positive: scene engine

**Verdict: standing scene.** Imagination in the mechanism: a model of the whole scene exists before the question is read, and the answer is produced from it.

- composition accuracy 1.00 (threshold 0.9)
- whole scene decodable before the question at layer 1 (probe 1.00)
- transplanting that scene site makes the answer follow the other world 100% of the time (layer 1)

| behaviour | value |
|---|---|
| accuracy (5-letter worlds) | 1.000 |
| questions needing composition | 1.000 |
| questions about stated pairs | 1.000 |
| same answer under 4 reshuffles | 1.000 |
| 7-letter worlds (never trained) | 1.000 |

Linear probes for rank (chance 0.20). *scene* = after the premises, before the question.

| layer | site | probe | untrained copy | surface counts |
|---|---|---|---|---|
| 1 | X | **1.00** | n/a | 0.59 |
| 1 | Y | **1.00** | n/a | 0.60 |
| 1 | scene | **1.00** | n/a | 0.60 |

Transplant a whole site from a world with the opposite answer: share of answers that follow it.

| layer | scene | X | Y | ANS |
|---|---|---|---|---|
| 1 | 1.00 | 0.50 | 0.53 | 1.00 |

Move only x (y keeps its rank). Transplant at x's site: everything, only the probe's rank subspace, or a random subspace of equal size.

| layer | everything | rank subspace | random subspace | dims |
|---|---|---|---|---|
| 1 | 1.00 | **1.00** | 0.73 | 4 |

## known negative: surface heuristic

**Verdict: no imagination.** No working imagination: questions that need premises combined are not answered, so behaviour is explained by stated pairs and surface statistics.

- composition accuracy 0.50 (threshold 0.9)

| behaviour | value |
|---|---|
| accuracy (5-letter worlds) | 0.956 |
| questions needing composition | 0.496 |
| questions about stated pairs | 1.000 |
| same answer under 4 reshuffles | 0.937 |
| 7-letter worlds (never trained) | 0.848 |

Linear probes for rank (chance 0.20). *scene* = after the premises, before the question.

| layer | site | probe | untrained copy | surface counts |
|---|---|---|---|---|
| 1 | X | **0.59** | n/a | 0.59 |
| 1 | Y | **0.60** | n/a | 0.60 |
| 1 | scene | **0.60** | n/a | 0.60 |

Transplant a whole site from a world with the opposite answer: share of answers that follow it.

| layer | scene | X | Y | ANS |
|---|---|---|---|---|
| 1 | 0.95 | 0.24 | 0.28 | 0.95 |

Move only x (y keeps its rank). Transplant at x's site: everything, only the probe's rank subspace, or a random subspace of equal size.

| layer | everything | rank subspace | random subspace | dims |
|---|---|---|---|---|
| 1 | 0.44 | **0.42** | 0.43 | 3 |

## plain transformer

**Verdict: something else.** Something else: the questions are answered correctly, but no scene representation that this detector can find is used (it may compare relations directly, or encode positions non-linearly).

- composition accuracy 0.91 (threshold 0.9)
- no site carries decodable rank information beyond the controls

Nothing in this architecture reserves a place for a scene; whatever is found here was invented by training.

| behaviour | value |
|---|---|
| accuracy (5-letter worlds) | 0.989 |
| questions needing composition | 0.905 |
| questions about stated pairs | 0.996 |
| same answer under 4 reshuffles | 0.983 |
| 7-letter worlds (never trained) | 0.860 |

Linear probes for rank (chance 0.20). *scene* = after the premises, before the question.

| layer | site | probe | untrained copy | surface counts |
|---|---|---|---|---|
| 0 | X | **0.20** | 0.19 | 0.59 |
| 0 | Y | **0.20** | 0.20 | 0.60 |
| 0 | scene | **0.21** | 0.21 | 0.60 |
| 1 | X | **0.29** | 0.21 | 0.59 |
| 1 | Y | **0.27** | 0.20 | 0.60 |
| 1 | scene | **0.33** | 0.31 | 0.60 |
| 2 | X | **0.69** | 0.20 | 0.59 |
| 2 | Y | **0.56** | 0.19 | 0.60 |
| 2 | scene | **0.37** | 0.32 | 0.60 |
| 3 | X | **0.69** | 0.19 | 0.59 |
| 3 | Y | **0.57** | 0.18 | 0.60 |
| 3 | scene | **0.41** | 0.32 | 0.60 |
| 4 | X | **0.68** | 0.20 | 0.59 |
| 4 | Y | **0.58** | 0.19 | 0.60 |
| 4 | scene | **0.40** | 0.32 | 0.60 |

Transplant a whole site from a world with the opposite answer: share of answers that follow it.

| layer | scene | X | Y | ANS |
|---|---|---|---|---|
| 0 | 0.01 | 0.01 | 0.01 | 0.01 |
| 1 | 0.01 | 0.06 | 0.04 | 0.01 |
| 2 | 0.01 | 0.42 | 0.61 | 0.01 |
| 3 | 0.01 | 0.01 | 0.01 | 0.99 |
| 4 | 0.01 | 0.01 | 0.01 | 0.99 |

Move only x (y keeps its rank). Transplant at x's site: everything, only the probe's rank subspace, or a random subspace of equal size.

| layer | everything | rank subspace | random subspace | dims |
|---|---|---|---|---|
| 0 | 0.01 | **0.01** | 0.01 | 4 |
| 1 | 0.01 | **0.01** | 0.01 | 4 |
| 2 | 0.81 | **0.02** | 0.03 | 4 |
| 3 | 0.01 | **0.01** | 0.01 | 4 |
| 4 | 0.01 | **0.01** | 0.01 | 4 |

## plain transformer, packed premises (learned a positional shortcut)

**Verdict: no imagination.** No working imagination: questions that need premises combined are not answered, so behaviour is explained by stated pairs and surface statistics.

- composition accuracy 0.58 (threshold 0.9)

| behaviour | value |
|---|---|
| accuracy (5-letter worlds) | 0.649 |
| questions needing composition | 0.576 |
| questions about stated pairs | 0.654 |
| same answer under 4 reshuffles | 0.233 |
| 7-letter worlds (never trained) | 0.547 |

Linear probes for rank (chance 0.20). *scene* = after the premises, before the question.

| layer | site | probe | untrained copy | surface counts |
|---|---|---|---|---|
| 0 | X | **0.19** | 0.19 | 0.59 |
| 0 | Y | **0.20** | 0.20 | 0.60 |
| 0 | scene | **0.21** | 0.21 | 0.60 |
| 1 | X | **0.26** | 0.21 | 0.59 |
| 1 | Y | **0.26** | 0.20 | 0.60 |
| 1 | scene | **0.26** | 0.33 | 0.60 |
| 2 | X | **0.27** | 0.18 | 0.59 |
| 2 | Y | **0.28** | 0.20 | 0.60 |
| 2 | scene | **0.26** | 0.34 | 0.60 |
| 3 | X | **0.29** | 0.20 | 0.59 |
| 3 | Y | **0.30** | 0.19 | 0.60 |
| 3 | scene | **0.24** | 0.34 | 0.60 |
| 4 | X | **0.27** | 0.20 | 0.59 |
| 4 | Y | **0.31** | 0.19 | 0.60 |
| 4 | scene | **0.23** | 0.33 | 0.60 |

Transplant a whole site from a world with the opposite answer: share of answers that follow it.

| layer | scene | X | Y | ANS |
|---|---|---|---|---|
| 0 | 0.32 | 0.32 | 0.32 | 0.32 |
| 1 | 0.32 | 0.50 | 0.43 | 0.41 |
| 2 | 0.32 | 0.46 | 0.47 | 0.42 |
| 3 | 0.32 | 0.33 | 0.35 | 0.62 |
| 4 | 0.32 | 0.32 | 0.32 | 0.65 |

Move only x (y keeps its rank). Transplant at x's site: everything, only the probe's rank subspace, or a random subspace of equal size.

| layer | everything | rank subspace | random subspace | dims |
|---|---|---|---|---|
| 0 | 0.35 | **0.35** | 0.35 | 4 |
| 1 | 0.52 | **0.35** | 0.35 | 4 |
| 2 | 0.49 | **0.35** | 0.35 | 4 |
| 3 | 0.36 | **0.35** | 0.34 | 4 |
| 4 | 0.35 | **0.35** | 0.35 | 4 |
