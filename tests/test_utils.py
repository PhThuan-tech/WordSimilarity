from __future__ import annotations

import numpy as np

from src.common.datasets import LabeledWordPair
from src.common.embeddings import EmbeddingStore
from src.common.utils import get_coverage


def test_coverage_counts_pairs_unique_words_and_occurrences() -> None:
    embeddings = EmbeddingStore(["a", "b"], np.asarray([[1, 0], [0, 1]]))
    pairs = [
        LabeledWordPair("a", "b", "SYN"),
        LabeledWordPair("a", "c", "ANT"),
        LabeledWordPair("c", "d", "ANT"),
    ]

    report = get_coverage(pairs, embeddings)

    assert report.total_pairs == 3
    assert report.covered_pairs == 1
    assert report.oov_pairs == 2
    assert report.total_unique_words == 4
    assert report.covered_unique_words == 2
    assert report.oov_unique_words == ("c", "d")
    assert report.oov_occurrences == {"c": 2, "d": 1}
    assert report.pair_coverage == 1 / 3
    assert report.word_coverage == 0.5
