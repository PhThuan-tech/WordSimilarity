from __future__ import annotations

import numpy as np
import pytest

from src.common import EmbeddingStore, LabeledWordPair
from src.task3.features import FeatureError, build_feature_matrix, pair_features


def test_pair_features_are_symmetric_finite_and_have_expected_shape() -> None:
    first = np.asarray([1.0, 0.0], dtype=np.float32)
    second = np.asarray([0.0, 2.0], dtype=np.float32)

    forward = pair_features(first, second)
    reverse = pair_features(second, first)

    assert forward.shape == (5,)
    assert np.isfinite(forward).all()
    np.testing.assert_allclose(forward, reverse)
    np.testing.assert_allclose(forward, [1.0, 2.0, 0.0, 0.0, 0.0])


def test_pair_features_reject_zero_norm_and_mismatched_shapes() -> None:
    with pytest.raises(FeatureError, match="zero-norm"):
        pair_features(np.asarray([0.0, 0.0]), np.asarray([1.0, 0.0]))
    with pytest.raises(FeatureError, match="same shape"):
        pair_features(np.asarray([1.0]), np.asarray([1.0, 2.0]))


def test_build_feature_matrix_records_oov_without_zero_imputation() -> None:
    embeddings = EmbeddingStore(
        ["đẹp", "xinh"],
        np.asarray([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32),
    )
    pairs = [
        LabeledWordPair("đẹp", "xinh", "SYN"),
        LabeledWordPair("đẹp", "xấu", "ANT"),
    ]

    batch = build_feature_matrix(pairs, embeddings)

    assert batch.matrix.shape == (1, 5)
    assert batch.labels.tolist() == ["SYN"]
    assert batch.source_indices == (0,)
    assert batch.oov_by_index == {1: ("xấu",)}
    assert batch.covered_rows == 1
    assert batch.oov_rows == 1
