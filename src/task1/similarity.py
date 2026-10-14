"""Cosine predictions and covered-subset evaluation for ViSim."""

from dataclasses import asdict

import numpy as np
from scipy.stats import pearsonr, spearmanr

from src.common import get_coverage


def cosine_similarity(vector1: np.ndarray, vector2: np.ndarray) -> float:
    """Return cosine; reject invalid shapes, nonfinite values and zero norms."""
    a = np.asarray(vector1, dtype=np.float64)
    b = np.asarray(vector2, dtype=np.float64)
    if a.ndim != 1 or a.shape != b.shape or not a.size:
        raise ValueError("Vectors must be nonempty, one-dimensional and equal in shape.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Vectors must contain only finite values.")
    # Scale before normalization to avoid overflow/underflow for finite vectors.
    a_scale, b_scale = np.max(np.abs(a)), np.max(np.abs(b))
    if a_scale == 0 or b_scale == 0:
        raise ValueError("Cosine is undefined for a zero-norm vector.")
    a, b = a / a_scale, b / b_scale
    return float(np.clip(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b)), -1, 1))


def _correlations(predictions, gold):
    n = len(predictions)
    reason = None
    if n < 2:
        reason = "fewer_than_two_samples"
    elif np.ptp(predictions) == 0 or np.ptp(gold) == 0:
        reason = "constant_input"
    if reason:
        return {"n": n, "pearson": None, "spearman": None, "undefined_reason": reason}
    return {
        "n": n,
        "pearson": float(pearsonr(predictions, gold).statistic),
        "spearman": float(spearmanr(predictions, gold).statistic),
        "undefined_reason": None,
    }


def evaluate_visim(embeddings, pairs) -> tuple[list[dict], dict]:
    """Keep every input row; evaluate both gold columns on finite predictions."""
    pairs = list(pairs)
    coverage = get_coverage(pairs, embeddings)
    rows = []
    for pair in pairs:
        missing = sorted({w for w in (pair.word1, pair.word2) if w not in embeddings})
        row = asdict(pair)
        row.update(cosine=None, status="oov" if missing else "covered",
                   oov_words=";".join(missing))
        if not missing:
            a = embeddings.get_vector(pair.word1)
            b = embeddings.get_vector(pair.word2)
            if not np.any(a) or not np.any(b):
                row["status"] = "zero_norm"
            else:
                row["cosine"] = cosine_similarity(a, b)
        rows.append(row)
    evaluated = [r for r in rows if r["cosine"] is not None and np.isfinite(r["cosine"])]
    predictions = [r["cosine"] for r in evaluated]
    summary = {
        "coverage": {**asdict(coverage), "pair_coverage": coverage.pair_coverage,
                     "word_coverage": coverage.word_coverage},
        "evaluated_pairs": len(evaluated),
        "zero_norm_pairs": sum(r["status"] == "zero_norm" for r in rows),
        "gold_columns": ["sim1", "sim2"],
        "metrics": {key: _correlations(predictions, [r[key] for r in evaluated])
                    for key in ("sim1", "sim2")},
        "evaluation_policy": "Only finite cosine predictions; exclude OOV and zero-norm pairs.",
    }
    return rows, summary
