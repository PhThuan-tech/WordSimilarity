
import numpy as np
import pytest

from src.task2.nearest import (
    NearestWordSearcher,
)
from src.common import OutOfVocabularyError


class FakeEmbeddings:
    def __init__(self, vocab, matrix):
        self.vocab = vocab
        self.matrix = np.asarray(matrix, dtype=np.float32)

    def get_vocabulary(self):
        return self.vocab

    def get_embedding_matrix(self):
        return self.matrix

    def contains(self, word):
        return word in self.vocab


@pytest.fixture
def embeddings():
    return FakeEmbeddings(
        ["sinh_viên", "học_sinh", "giáo_viên", "mèo"],
        [
            [1, 0],
            [0.9, 0.1],
            [0, 1],
            [-1, 0],
        ],
    )


def test_top_k_order_and_similarity(embeddings):
    searcher = NearestWordSearcher(embeddings)
    result = searcher.find_nearest("sinh_viên", 2)

    assert [item.word for item in result] == [
        "học_sinh", "giáo_viên"
    ]
    assert result[0].similarity > result[1].similarity


def test_query_is_excluded(embeddings):
    result = NearestWordSearcher(embeddings).find_nearest(
        "sinh_viên", 3
    )
    assert all(item.word != "sinh_viên" for item in result)


def test_k_larger_than_vocabulary(embeddings):
    result = NearestWordSearcher(embeddings).find_nearest(
        "sinh_viên", 100
    )
    assert len(result) == len(embeddings.vocab) - 1


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(embeddings, k):
    with pytest.raises(ValueError):
        NearestWordSearcher(embeddings).find_nearest(
            "sinh_viên", k
        )


def test_oov_raises_exception(embeddings):
    with pytest.raises(OutOfVocabularyError):
        NearestWordSearcher(embeddings).find_nearest(
            "khong_co_trong_vocab", 2
        )


def test_zero_vector_does_not_create_nan():
    emb = FakeEmbeddings(
        ["a", "b", "c"],
        [[1, 0], [0, 0], [0, 1]],
    )
    result = NearestWordSearcher(emb).find_nearest("a", 2)

    assert all(np.isfinite(item.similarity) for item in result)
