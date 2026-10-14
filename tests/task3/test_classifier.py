from __future__ import annotations

import numpy as np

from src.common import LabeledWordPair
from src.task3.classifier import (
    audit_overlap,
    create_classifier,
    evaluate_predictions,
    train_classifier,
)


def test_overlap_audit_recognizes_reversed_and_ambiguous_pairs() -> None:
    train = [
        LabeledWordPair("cao", "thấp", "ANT"),
        LabeledWordPair("xinh", "đẹp", "SYN"),
        LabeledWordPair("đẹp", "xinh", "ANT"),
    ]
    test = [
        LabeledWordPair("thấp", "cao", "ANT"),
        LabeledWordPair("đẹp", "xinh", "SYN"),
        LabeledWordPair("mới", "cũ", "ANT"),
    ]

    audit = audit_overlap(train, test)

    assert audit.overlapping_unique_pairs == 2
    assert audit.overlapping_test_rows == 2
    assert audit.matching_label_rows == 2
    assert audit.mismatching_label_rows == 0
    assert audit.ambiguous_train_label_rows == 1
    assert audit.train_conflicting_unique_pairs == 1


def test_classifier_is_a_class_balanced_seeded_pipeline() -> None:
    pipeline = create_classifier(random_state=17)
    classifier = pipeline.named_steps["classifier"]

    assert list(pipeline.named_steps) == ["scaler", "classifier"]
    assert classifier.class_weight == "balanced"
    assert classifier.random_state == 17


def test_training_is_reproducible_for_a_fixed_seed() -> None:
    features = np.asarray(
        [
            [0.0, 1.0],
            [0.1, 0.9],
            [0.2, 0.8],
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(["SYN", "SYN", "SYN", "ANT", "ANT", "ANT"])

    first = train_classifier(features, labels, random_state=42)
    second = train_classifier(features, labels, random_state=42)

    np.testing.assert_array_equal(first.predict(features), second.predict(features))
    np.testing.assert_allclose(
        first.predict_proba(features), second.predict_proba(features)
    )


def test_evaluation_metrics_match_known_confusion_matrix() -> None:
    gold = np.asarray(["ANT", "ANT", "SYN", "SYN"])
    predicted = np.asarray(["ANT", "SYN", "SYN", "SYN"])

    metrics = evaluate_predictions(gold, predicted)

    assert metrics["evaluated_rows"] == 4
    assert metrics["label_order"] == ["ANT", "SYN"]
    assert metrics["positive_class"] == "SYN"
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]
    assert metrics["accuracy"] == 0.75
    assert metrics["per_class"]["SYN"]["precision"] == 2 / 3
    assert metrics["per_class"]["SYN"]["recall"] == 1.0
