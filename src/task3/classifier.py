"""Leakage audit, Logistic Regression baseline, and evaluation metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.common import LabeledWordPair


LABEL_ORDER = ("ANT", "SYN")
POSITIVE_CLASS = "SYN"
DEFAULT_RANDOM_STATE = 42


@dataclass(frozen=True, slots=True)
class OverlapAudit:
    train_rows: int
    test_rows: int
    unique_train_pairs: int
    unique_test_pairs: int
    overlapping_unique_pairs: int
    overlapping_test_rows: int
    matching_label_rows: int
    mismatching_label_rows: int
    ambiguous_train_label_rows: int
    train_conflicting_unique_pairs: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def canonical_pair(word1: str, word2: str) -> tuple[str, str]:
    """Canonicalize a pair so reversed word order has the same key."""

    return (word1, word2) if word1 <= word2 else (word2, word1)


def audit_overlap(
    train_pairs: Sequence[LabeledWordPair],
    test_pairs: Sequence[LabeledWordPair],
) -> OverlapAudit:
    """Describe unordered train/test overlap without influencing predictions."""

    train_labels: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in train_pairs:
        train_labels[canonical_pair(pair.word1, pair.word2)].add(pair.relation)

    test_keys = {canonical_pair(pair.word1, pair.word2) for pair in test_pairs}
    overlapping_keys = test_keys.intersection(train_labels)
    overlapping_rows = 0
    matching_rows = 0
    mismatching_rows = 0
    ambiguous_rows = 0

    for pair in test_pairs:
        labels = train_labels.get(canonical_pair(pair.word1, pair.word2))
        if not labels:
            continue
        overlapping_rows += 1
        if pair.relation in labels:
            matching_rows += 1
        else:
            mismatching_rows += 1
        if len(labels) > 1:
            ambiguous_rows += 1

    return OverlapAudit(
        train_rows=len(train_pairs),
        test_rows=len(test_pairs),
        unique_train_pairs=len(train_labels),
        unique_test_pairs=len(test_keys),
        overlapping_unique_pairs=len(overlapping_keys),
        overlapping_test_rows=overlapping_rows,
        matching_label_rows=matching_rows,
        mismatching_label_rows=mismatching_rows,
        ambiguous_train_label_rows=ambiguous_rows,
        train_conflicting_unique_pairs=sum(
            len(labels) > 1 for labels in train_labels.values()
        ),
    )


def create_classifier(
    *, random_state: int = DEFAULT_RANDOM_STATE
) -> Pipeline:
    """Create the deterministic, class-balanced baseline pipeline."""

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_classifier(
    features: NDArray[np.floating],
    labels: NDArray[np.str_] | Sequence[str],
    *,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Pipeline:
    """Fit preprocessing and model on training data only."""

    matrix = np.asarray(features)
    target = np.asarray(labels, dtype=np.str_)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("Training features must be a non-empty 2D matrix.")
    if target.ndim != 1 or target.shape[0] != matrix.shape[0]:
        raise ValueError("Training labels must align with feature rows.")
    unknown_labels = set(target.tolist()).difference(LABEL_ORDER)
    if unknown_labels:
        raise ValueError(f"Unsupported training labels: {sorted(unknown_labels)}")
    if len(np.unique(target)) != 2:
        raise ValueError("Training data must contain both ANT and SYN labels.")

    model = create_classifier(random_state=random_state)
    model.fit(matrix, target)
    return model


def evaluate_predictions(
    gold: NDArray[np.str_] | Sequence[str],
    predicted: NDArray[np.str_] | Sequence[str],
) -> dict[str, object]:
    """Return JSON-serializable metrics using the documented label order."""

    gold_array = np.asarray(gold, dtype=np.str_)
    predicted_array = np.asarray(predicted, dtype=np.str_)
    if gold_array.ndim != 1 or predicted_array.ndim != 1:
        raise ValueError("Gold and predicted labels must be one-dimensional.")
    if gold_array.shape != predicted_array.shape or gold_array.size == 0:
        raise ValueError("Gold and predicted labels must be non-empty and aligned.")
    unsupported = set(gold_array.tolist() + predicted_array.tolist()).difference(
        LABEL_ORDER
    )
    if unsupported:
        raise ValueError(f"Unsupported evaluation labels: {sorted(unsupported)}")

    report = classification_report(
        gold_array,
        predicted_array,
        labels=list(LABEL_ORDER),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        gold_array, predicted_array, labels=list(LABEL_ORDER)
    ).tolist()
    return {
        "evaluated_rows": int(gold_array.size),
        "label_order": list(LABEL_ORDER),
        "positive_class": POSITIVE_CLASS,
        "confusion_matrix": matrix,
        "accuracy": float(accuracy_score(gold_array, predicted_array)),
        "per_class": {
            label: {
                "precision": float(report[label]["precision"]),
                "recall": float(report[label]["recall"]),
                "f1": float(report[label]["f1-score"]),
                "support": int(report[label]["support"]),
            }
            for label in LABEL_ORDER
        },
        "macro_avg": {
            "precision": float(report["macro avg"]["precision"]),
            "recall": float(report["macro avg"]["recall"]),
            "f1": float(report["macro avg"]["f1-score"]),
            "support": int(report["macro avg"]["support"]),
        },
        "weighted_avg": {
            "precision": float(report["weighted avg"]["precision"]),
            "recall": float(report["weighted avg"]["recall"]),
            "f1": float(report["weighted avg"]["f1-score"]),
            "support": int(report["weighted avg"]["support"]),
        },
    }


def synonym_probabilities(
    model: Pipeline,
    features: NDArray[np.floating],
) -> NDArray[np.float64]:
    """Return the probability of SYN using the model's learned class order."""

    classes = model.named_steps["classifier"].classes_
    matches = np.flatnonzero(classes == POSITIVE_CLASS)
    if matches.size != 1:
        raise RuntimeError("Trained classifier does not expose a unique SYN class.")
    return model.predict_proba(features)[:, int(matches[0])]
