"""Symmetric Word2Vec feature engineering for synonym/antonym pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.common import EmbeddingStore, LabeledWordPair


class FeatureError(ValueError):
    """Raised when a pair cannot produce a valid numeric feature vector."""


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    """Features plus row mapping back to the original pair collection."""

    matrix: NDArray[np.float32]
    labels: NDArray[np.str_]
    source_indices: tuple[int, ...]
    oov_by_index: dict[int, tuple[str, ...]]

    @property
    def covered_rows(self) -> int:
        return len(self.source_indices)

    @property
    def oov_rows(self) -> int:
        return len(self.oov_by_index)


def pair_features(
    vector1: NDArray[np.floating],
    vector2: NDArray[np.floating],
) -> NDArray[np.float32]:
    """Return ``[abs(v1-v2), v1*v2, cosine(v1,v2)]``.

    The representation is symmetric: swapping the two input words produces the
    same feature vector. Invalid shapes, non-finite values and zero-norm vectors
    fail explicitly instead of creating misleading zero/NaN features.
    """

    first = np.asarray(vector1, dtype=np.float32)
    second = np.asarray(vector2, dtype=np.float32)
    if first.ndim != 1 or second.ndim != 1:
        raise FeatureError("Word vectors must be one-dimensional.")
    if first.shape != second.shape:
        raise FeatureError(
            f"Word vectors must have the same shape, got {first.shape} and {second.shape}."
        )
    if first.size == 0:
        raise FeatureError("Word vectors must not be empty.")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise FeatureError("Word vectors must contain only finite values.")

    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm == 0.0 or second_norm == 0.0:
        raise FeatureError("Cosine similarity is undefined for a zero-norm vector.")

    cosine = float(np.dot(first, second) / (first_norm * second_norm))
    # Floating-point roundoff can push cosine infinitesimally outside [-1, 1].
    cosine = float(np.clip(cosine, -1.0, 1.0))
    features = np.concatenate(
        (
            np.abs(first - second),
            first * second,
            np.asarray([cosine], dtype=np.float32),
        )
    ).astype(np.float32, copy=False)
    if not np.isfinite(features).all():
        raise FeatureError("Engineered features contain a non-finite value.")
    return features


def build_feature_matrix(
    pairs: Sequence[LabeledWordPair],
    embeddings: EmbeddingStore,
) -> FeatureBatch:
    """Build features for covered pairs and record OOV rows without imputation."""

    feature_rows: list[NDArray[np.float32]] = []
    labels: list[str] = []
    source_indices: list[int] = []
    oov_by_index: dict[int, tuple[str, ...]] = {}

    for index, pair in enumerate(pairs):
        missing = tuple(
            dict.fromkeys(
                word
                for word in (pair.word1, pair.word2)
                if not embeddings.contains(word)
            )
        )
        if missing:
            oov_by_index[index] = missing
            continue

        try:
            features = pair_features(
                embeddings.get_vector(pair.word1),
                embeddings.get_vector(pair.word2),
            )
        except FeatureError as exc:
            raise FeatureError(
                f"Cannot build features for pair at index {index} "
                f"({pair.word1!r}, {pair.word2!r}): {exc}"
            ) from exc
        feature_rows.append(features)
        labels.append(pair.relation)
        source_indices.append(index)

    feature_dimension = embeddings.dimension * 2 + 1
    if feature_rows:
        matrix = np.stack(feature_rows).astype(np.float32, copy=False)
    else:
        matrix = np.empty((0, feature_dimension), dtype=np.float32)

    return FeatureBatch(
        matrix=matrix,
        labels=np.asarray(labels, dtype=np.str_),
        source_indices=tuple(source_indices),
        oov_by_index=oov_by_index,
    )
