
import io
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest

import src.task2.cli as task2_cli
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
    assert all(item.word != "b" for item in result)


def test_zero_norm_query_is_rejected():
    emb = FakeEmbeddings(["zero", "a"], [[0, 0], [1, 0]])

    with pytest.raises(ValueError, match="norm bằng 0"):
        NearestWordSearcher(emb).find_nearest("zero", 1)


def test_tie_break_is_alphabetical_at_top_k_boundary():
    emb = FakeEmbeddings(
        ["query", "e", "d", "c", "b", "a"],
        [[1, 0], [0, 1], [0, 1], [0, 1], [0, 1], [0, 1]],
    )

    result = NearestWordSearcher(emb).find_nearest("query", 2)

    assert [item.word for item in result] == ["a", "b"]


def test_cli_uses_requested_embedding_path_and_emits_utf8(monkeypatch):
    emb = FakeEmbeddings(
        ["sinh_viên", "học_sinh"],
        [[1, 0], [0.9, 0.1]],
    )
    requested_path = Path("custom-vectors.txt")
    captured_paths = []
    monkeypatch.setattr(
        task2_cli,
        "load_embeddings",
        lambda path: captured_paths.append(path) or emb,
    )

    raw_output = io.BytesIO()
    legacy_stream = io.TextIOWrapper(raw_output, encoding="cp1252")
    monkeypatch.setattr(task2_cli.sys, "stdout", legacy_stream)

    result = task2_cli.run(
        Namespace(word="sinh_viên", k=1, embeddings=requested_path)
    )
    legacy_stream.flush()
    output = raw_output.getvalue().decode("utf-8")

    assert result == 0
    assert captured_paths == [requested_path]
    assert "Đang tải dữ liệu Word2Vec" in output
    assert "học_sinh" in output
