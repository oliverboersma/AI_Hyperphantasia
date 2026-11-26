# Imagination detector report

Verdict per subject, from behaviour, linear probes, and interchange interventions. See docs/IMAGINATION_DETECTOR.md for the method.

| subject | verdict |
|---|---|
| known positive: scene engine | standing scene |
| known negative: surface heuristic | no imagination |
| plain transformer | no imagination |

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

**Verdict: no imagination.** No working imagination: questions that need premises combined are not answered, so behaviour is explained by stated pairs and surface statistics.

- composition accuracy 0.83 (threshold 0.9)

Nothing in this architecture reserves a place for a scene; whatever is found here was invented by training.

| behaviour | value |
|---|---|
| accuracy (5-letter worlds) | 0.940 |
| questions needing composition | 0.833 |
| questions about stated pairs | 0.886 |
| same answer under 4 reshuffles | 0.838 |
| 7-letter worlds (never trained) | 0.834 |

Linear probes for rank (chance 0.20). *scene* = after the premises, before the question.

| layer | site | probe | untrained copy | surface counts |
|---|---|---|---|---|
| 0 | X | **0.19** | 0.20 | 0.59 |
| 0 | Y | **0.19** | 0.20 | 0.60 |
| 0 | scene | **0.21** | 0.21 | 0.60 |
| 1 | X | **0.18** | 0.20 | 0.59 |
| 1 | Y | **0.19** | 0.21 | 0.60 |
| 1 | scene | **0.35** | 0.31 | 0.60 |
| 2 | X | **0.59** | 0.19 | 0.59 |
| 2 | Y | **0.62** | 0.20 | 0.60 |
| 2 | scene | **0.35** | 0.32 | 0.60 |
| 3 | X | **0.69** | 0.20 | 0.59 |
| 3 | Y | **0.71** | 0.20 | 0.60 |
| 3 | scene | **0.37** | 0.32 | 0.60 |
| 4 | X | **0.68** | 0.19 | 0.59 |
| 4 | Y | **0.72** | 0.20 | 0.60 |
| 4 | scene | **0.35** | 0.32 | 0.60 |

Transplant a whole site from a world with the opposite answer: share of answers that follow it.

| layer | scene | X | Y | ANS |
|---|---|---|---|---|
| 0 | 0.08 | 0.08 | 0.08 | 0.08 |
| 1 | 0.07 | 0.08 | 0.07 | 0.07 |
| 2 | 0.08 | 0.52 | 0.47 | 0.07 |
| 3 | 0.07 | 0.49 | 0.53 | 0.08 |
| 4 | 0.08 | 0.08 | 0.08 | 0.93 |

Move only x (y keeps its rank). Transplant at x's site: everything, only the probe's rank subspace, or a random subspace of equal size.

| layer | everything | rank subspace | random subspace | dims |
|---|---|---|---|---|
| 0 | 0.11 | **0.11** | 0.11 | 4 |
| 1 | 0.11 | **0.12** | 0.11 | 4 |
| 2 | 0.86 | **0.83** | 0.12 | 4 |
| 3 | 0.87 | **0.17** | 0.20 | 4 |
| 4 | 0.11 | **0.11** | 0.11 | 4 |
