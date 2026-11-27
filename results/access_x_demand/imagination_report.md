# Imagination detector report

Verdict per subject, from behaviour, linear probes, and interchange interventions. See docs/IMAGINATION_DETECTOR.md for the method.

| subject | verdict |
|---|---|
| known positive: scene engine | standing scene |
| known negative: surface heuristic | no imagination |
| plain transformer, mixed left-of and position questions | no imagination |
| Phantasm Slots (side door, examined shut), mixed left-of and position questions | no imagination |

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

## plain transformer, mixed left-of and position questions

**Verdict: no imagination.** No working imagination: questions that need premises combined are not answered, so behaviour is explained by stated pairs and surface statistics.

- composition accuracy 0.75 (threshold 0.9)

| behaviour | value |
|---|---|
| accuracy (5-letter worlds) | 0.967 |
| questions needing composition | 0.750 |
| questions about stated pairs | 0.974 |
| same answer under 4 reshuffles | 0.902 |
| 7-letter worlds (never trained) | 0.821 |

Linear probes for rank (chance 0.20). *scene* = after the premises, before the question.

| layer | site | probe | untrained copy | surface counts |
|---|---|---|---|---|
| 0 | X | **0.19** | 0.19 | 0.59 |
| 0 | Y | **0.20** | 0.20 | 0.60 |
| 0 | scene | **0.21** | 0.21 | 0.60 |
| 1 | X | **0.19** | 0.21 | 0.59 |
| 1 | Y | **0.18** | 0.20 | 0.60 |
| 1 | scene | **0.35** | 0.31 | 0.60 |
| 2 | X | **0.58** | 0.20 | 0.59 |
| 2 | Y | **0.59** | 0.19 | 0.60 |
| 2 | scene | **0.39** | 0.32 | 0.60 |
| 3 | X | **0.68** | 0.19 | 0.59 |
| 3 | Y | **0.61** | 0.18 | 0.60 |
| 3 | scene | **0.39** | 0.32 | 0.60 |
| 4 | X | **0.67** | 0.20 | 0.59 |
| 4 | Y | **0.63** | 0.19 | 0.60 |
| 4 | scene | **0.37** | 0.32 | 0.60 |

Transplant a whole site from a world with the opposite answer: share of answers that follow it.

| layer | scene | X | Y | ANS |
|---|---|---|---|---|
| 0 | 0.04 | 0.04 | 0.04 | 0.04 |
| 1 | 0.04 | 0.11 | 0.12 | 0.07 |
| 2 | 0.04 | 0.47 | 0.51 | 0.06 |
| 3 | 0.04 | 0.42 | 0.07 | 0.59 |
| 4 | 0.04 | 0.04 | 0.04 | 0.95 |

Move only x (y keeps its rank). Transplant at x's site: everything, only the probe's rank subspace, or a random subspace of equal size.

| layer | everything | rank subspace | random subspace | dims |
|---|---|---|---|---|
| 0 | 0.05 | **0.05** | 0.05 | 4 |
| 1 | 0.16 | **0.05** | 0.06 | 4 |
| 2 | 0.86 | **0.06** | 0.09 | 4 |
| 3 | 0.82 | **0.06** | 0.10 | 4 |
| 4 | 0.05 | **0.05** | 0.05 | 4 |

## Phantasm Slots (side door, examined shut), mixed left-of and position questions

**Verdict: no imagination.** No working imagination: questions that need premises combined are not answered, so behaviour is explained by stated pairs and surface statistics.

- composition accuracy 0.83 (threshold 0.9)

| behaviour | value |
|---|---|
| accuracy (5-letter worlds) | 0.858 |
| questions needing composition | 0.833 |
| questions about stated pairs | 0.835 |
| same answer under 4 reshuffles | 0.898 |
| 7-letter worlds (never trained) | 0.659 |

Linear probes for rank (chance 0.20). *scene* = after the premises, before the question.

| layer | site | probe | untrained copy | surface counts |
|---|---|---|---|---|
| 0 | X | **0.20** | 0.19 | 0.59 |
| 0 | Y | **0.19** | 0.20 | 0.60 |
| 0 | scene | **0.51** | 0.35 | 0.60 |
| 1 | X | **0.32** | 0.20 | 0.59 |
| 1 | Y | **0.28** | 0.20 | 0.60 |
| 1 | scene | **0.51** | 0.34 | 0.60 |
| 2 | X | **0.40** | 0.19 | 0.59 |
| 2 | Y | **0.34** | 0.21 | 0.60 |
| 2 | scene | **0.51** | 0.33 | 0.60 |
| 3 | X | **0.45** | 0.19 | 0.59 |
| 3 | Y | **0.42** | 0.20 | 0.60 |
| 3 | scene | **0.51** | 0.34 | 0.60 |
| 4 | X | **0.47** | 0.20 | 0.59 |
| 4 | Y | **0.42** | 0.20 | 0.60 |
| 4 | scene | **0.51** | 0.33 | 0.60 |

Transplant a whole site from a world with the opposite answer: share of answers that follow it.

| layer | scene | X | Y | ANS |
|---|---|---|---|---|
| 0 | 0.87 | 0.14 | 0.14 | 0.14 |
| 1 | 0.78 | 0.16 | 0.16 | 0.19 |
| 2 | 0.64 | 0.14 | 0.14 | 0.28 |
| 3 | 0.41 | 0.15 | 0.15 | 0.57 |
| 4 | 0.14 | 0.14 | 0.14 | 0.87 |

Move only x (y keeps its rank). Transplant at x's site: everything, only the probe's rank subspace, or a random subspace of equal size.

| layer | everything | rank subspace | random subspace | dims |
|---|---|---|---|---|
| 0 | 0.14 | **0.14** | 0.14 | 4 |
| 1 | 0.16 | **0.14** | 0.14 | 4 |
| 2 | 0.14 | **0.14** | 0.14 | 4 |
| 3 | 0.15 | **0.14** | 0.14 | 4 |
| 4 | 0.14 | **0.14** | 0.14 | 4 |
