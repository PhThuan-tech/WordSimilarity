# Task 3 — Synonym–Antonym Classification

## Protocol

- Features: `[abs(v1-v2), v1*v2, cosine(v1,v2)]`.
- Model: StandardScaler + class-balanced Logistic Regression.
- Random seed: `42`.
- Duplicate and conflicting valid training rows are retained.
- OOV rows remain in the CSV but are excluded from model metrics.
- The overlap audit is reporting-only; predictions never use label lookup.

## Data and coverage

| Split | Total | Covered | OOV | Coverage |
|---|---:|---:|---:|---:|
| Train | 13558 | 8004 | 5554 | 59.04% |
| ViCon test | 1400 | 1173 | 227 | 83.79% |

## Evaluation on covered ViCon rows

Evaluated rows: **1173**

| Label | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| ANT | 0.8019 | 0.7731 | 0.7873 | 639 |
| SYN | 0.7397 | 0.7715 | 0.7553 | 534 |
| Macro avg | 0.7708 | 0.7723 | 0.7713 | 1173 |

Accuracy: **0.7724**

Confusion matrix (rows = gold, columns = prediction; order ANT, SYN):

```text
[494, 145]
[122, 412]
```

## Data-quality and overlap audit

- Malformed training rows: 4.
- Ordered duplicate rows retained: 782.
- Unordered duplicate rows retained: 794.
- Conflicting unordered training pairs retained: 7.
- Overlapping unique train/test pairs: 1397.
- Overlapping ViCon rows: 1400.
- ViCon rows whose train pair has ambiguous labels: 4.

## Interpretation limit

ViCon substantially overlaps the supplied training pairs when word order is ignored. These required-protocol metrics therefore do not measure clean generalization to unseen pairs and must not be presented as such.
