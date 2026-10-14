# Task 1 — Cosine similarity and ViSim evaluation

Run from the repository root after installing `requirements.txt`:

```powershell
python main.py similarity --output results/task1_predictions.csv
python -m pytest -q
```

Optional `--embeddings` and `--dataset` arguments select other input files.
The command creates the output directory and writes a UTF-8 CSV plus
`results/task1_predictions_summary.json`. Algorithms live in `src/task1/`;
the shared loaders and other tasks are unchanged.

The CSV preserves all input rows and original gold scores. Status `covered`
means a cosine was computed; `oov` leaves cosine blank and lists missing words;
`zero_norm` leaves cosine blank for a vector whose norm is zero. No fallback
vectors are manufactured. Coverage counts vocabulary availability separately
from the number of usable predictions.

Pearson and Spearman use only finite predictions with their matching gold
scores. Both Sim1 and Sim2 are reported. Spearman measures rank agreement,
while Pearson measures linear agreement. Sim2 is a rescaling of Sim1; rounding
in the supplied scores causes a small difference in Pearson. Undefined
correlations (fewer than two samples or constant input) are JSON null with an
explicit reason, rather than NaN.

## Results on the supplied data

| Metric | Sim1 | Sim2 |
|---|---|---|
| Evaluated pairs | 344 | 344 |
| Pearson | 0.4468477985 | 0.4468197550 |
| Spearman | 0.4077568888 | 0.4077568888 |

There are 400 CSV rows: 344 covered (86%), 56 OOV, and no zero-norm pairs.
These correlations describe only the covered subset; they do not assess
predictions for the missing 56 pairs. Experiment outputs remain generated
files under `results/` and can be reproduced with the command above.

Tests cover cosine geometry and symmetry, invalid shapes, zero/nonfinite
vectors, extreme finite values, Unicode tokens, OOV exclusion, correlations
against SciPy, undefined metrics, repeatable CSV/JSON output, input overwrite
protection, and all 400 real ViSim rows.
