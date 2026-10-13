"""Task 3: Word2Vec-based synonym/antonym classification."""

from .classifier import (
    LABEL_ORDER,
    POSITIVE_CLASS,
    OverlapAudit,
    audit_overlap,
    create_classifier,
    evaluate_predictions,
    train_classifier,
)
from .features import FeatureBatch, FeatureError, build_feature_matrix, pair_features
from .workflow import Task3Result, run_task3

__all__ = [
    "FeatureBatch",
    "FeatureError",
    "LABEL_ORDER",
    "OverlapAudit",
    "POSITIVE_CLASS",
    "Task3Result",
    "audit_overlap",
    "build_feature_matrix",
    "create_classifier",
    "evaluate_predictions",
    "pair_features",
    "run_task3",
    "train_classifier",
]
