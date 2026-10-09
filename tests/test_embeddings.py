from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.common.embeddings import (
    EmbeddingFormatError,
    OutOfVocabularyError,
    load_embeddings,
)


def write_embedding(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_loads_two_line_header_and_vietnamese_tokens(tmp_path: Path) -> None:
    path = write_embedding(
        tmp_path / "tiny.txt",
        "3\n2\nsinh_viên 1.0 0.0\nhọc_sinh 0.0 1.0\nđẹp 0.5 0.5\n",
    )

    embeddings = load_embeddings(path)

    assert embeddings.size == 3
    assert embeddings.dimension == 2
    assert embeddings.contains("sinh_viên")
    assert "đẹp" in embeddings
    assert embeddings.get_vocabulary() == ("sinh_viên", "học_sinh", "đẹp")
    np.testing.assert_allclose(embeddings.get_vector("đẹp"), [0.5, 0.5])
    assert embeddings.get_embedding_matrix().shape == (3, 2)
    assert not embeddings.get_embedding_matrix().flags.writeable


def test_oov_raises_instead_of_returning_zero_vector(tmp_path: Path) -> None:
    path = write_embedding(tmp_path / "tiny.txt", "1\n2\ncó 1 2\n")
    embeddings = load_embeddings(path)

    with pytest.raises(OutOfVocabularyError, match="không_có"):
        embeddings.get_vector("không_có")


@pytest.mark.parametrize(
    "content, message",
    [
        ("2\n2\na 1 2\n", "declares 2 vectors"),
        ("1\n2\na 1\n", "expected 2"),
        ("2\n2\na 1 2\na 3 4\n", "duplicate token"),
        ("1\n2\na NaN 2\n", "non-finite"),
        ("1 2\na 1 2\n", "integer vocabulary size"),
    ],
)
def test_rejects_invalid_embedding_files(
    tmp_path: Path, content: str, message: str
) -> None:
    path = write_embedding(tmp_path / "bad.txt", content)

    with pytest.raises(EmbeddingFormatError, match=message):
        load_embeddings(path)
