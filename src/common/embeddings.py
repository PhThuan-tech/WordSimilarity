"""Validated loader and read-only accessors for the supplied Word2Vec file."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


class EmbeddingFormatError(ValueError):
    """Raised when an embedding file violates the expected data contract."""


class OutOfVocabularyError(KeyError):
    """Raised when a requested token is not present in the embedding vocabulary."""


class EmbeddingStore:
    """Immutable vocabulary metadata with a read-only dense embedding matrix.

    Instances should normally be constructed through :func:`load_embeddings`.
    Row ``i`` in ``matrix`` always corresponds to ``vocabulary[i]``.
    """

    def __init__(
        self,
        vocabulary: Sequence[str],
        matrix: NDArray[np.floating],
    ) -> None:
        words = tuple(vocabulary)
        array = np.asarray(matrix, dtype=np.float32)

        if array.ndim != 2:
            raise ValueError("Embedding matrix must be two-dimensional.")
        if len(words) != array.shape[0]:
            raise ValueError("Vocabulary size must equal the number of matrix rows.")
        if len(set(words)) != len(words):
            raise ValueError("Vocabulary must not contain duplicate tokens.")
        if not np.isfinite(array).all():
            raise ValueError("Embedding matrix contains NaN or infinite values.")

        # Own the data so callers cannot mutate it through an earlier reference.
        self._matrix = np.array(array, copy=True, order="C")
        self._matrix.flags.writeable = False
        self._vocabulary = words
        self._word_to_index = {word: index for index, word in enumerate(words)}

    @property
    def size(self) -> int:
        """Number of tokens in the vocabulary."""

        return len(self._vocabulary)

    @property
    def dimension(self) -> int:
        """Embedding dimension."""

        return int(self._matrix.shape[1])

    def __len__(self) -> int:
        return self.size

    def __contains__(self, word: object) -> bool:
        return isinstance(word, str) and word in self._word_to_index

    def __iter__(self) -> Iterator[str]:
        return iter(self._vocabulary)

    def contains(self, word: str) -> bool:
        """Return whether ``word`` has a pretrained vector."""

        return word in self._word_to_index

    def get_vector(self, word: str) -> NDArray[np.floating]:
        """Return a read-only view of a word vector.

        No fallback vector is manufactured for an unknown word.
        """

        try:
            index = self._word_to_index[word]
        except KeyError as exc:
            raise OutOfVocabularyError(f"Word is out of vocabulary: {word!r}") from exc
        vector = self._matrix[index]
        vector.flags.writeable = False
        return vector

    def get_vocabulary(self) -> tuple[str, ...]:
        """Return tokens in the same row order as the embedding matrix."""

        return self._vocabulary

    def get_embedding_matrix(self) -> NDArray[np.floating]:
        """Return the shared read-only matrix for vectorized computations."""

        return self._matrix


def _parse_positive_int(raw: str, name: str, path: Path, line_number: int) -> int:
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise EmbeddingFormatError(
            f"{path}: line {line_number} must contain an integer {name}."
        ) from exc
    if value <= 0:
        raise EmbeddingFormatError(
            f"{path}: line {line_number} {name} must be positive, got {value}."
        )
    return value


def load_embeddings(
    path: str | Path,
    *,
    dtype: np.dtype[np.floating] | type[np.floating] = np.float32,
) -> EmbeddingStore:
    """Load the assignment's two-line-header Word2Vec text format.

    Expected layout::

        <vocabulary size>\n
        <embedding dimension>\n
        <token> <value 1> ... <value d>\n

    Every row, declared count, duplicate token and numeric value is validated.
    Text is decoded strictly as UTF-8 (with an optional BOM on the first line).
    """

    embedding_path = Path(path)
    if not embedding_path.is_file():
        raise FileNotFoundError(f"Embedding file does not exist: {embedding_path}")

    requested_dtype = np.dtype(dtype)
    if requested_dtype.kind != "f":
        raise TypeError("Embedding dtype must be a floating-point dtype.")

    try:
        with embedding_path.open("r", encoding="utf-8-sig", newline=None) as handle:
            count_line = handle.readline()
            dimension_line = handle.readline()
            if not count_line or not dimension_line:
                raise EmbeddingFormatError(
                    f"{embedding_path}: expected two separate header lines."
                )

            declared_count = _parse_positive_int(
                count_line, "vocabulary size", embedding_path, 1
            )
            dimension = _parse_positive_int(
                dimension_line, "embedding dimension", embedding_path, 2
            )

            vocabulary: list[str] = []
            vectors: list[NDArray[np.floating]] = []
            seen: set[str] = set()

            for line_number, line in enumerate(handle, start=3):
                fields = line.split()
                if not fields:
                    raise EmbeddingFormatError(
                        f"{embedding_path}: blank embedding row at line {line_number}."
                    )
                if len(fields) != dimension + 1:
                    raise EmbeddingFormatError(
                        f"{embedding_path}: line {line_number} has "
                        f"{len(fields) - 1} values; expected {dimension}."
                    )

                word = fields[0]
                if word in seen:
                    raise EmbeddingFormatError(
                        f"{embedding_path}: duplicate token {word!r} at line {line_number}."
                    )
                try:
                    vector = np.asarray(fields[1:], dtype=requested_dtype)
                except ValueError as exc:
                    raise EmbeddingFormatError(
                        f"{embedding_path}: non-numeric value at line {line_number}."
                    ) from exc
                if not np.isfinite(vector).all():
                    raise EmbeddingFormatError(
                        f"{embedding_path}: non-finite value at line {line_number}."
                    )

                seen.add(word)
                vocabulary.append(word)
                vectors.append(vector)
    except UnicodeDecodeError as exc:
        raise EmbeddingFormatError(
            f"{embedding_path}: file is not valid UTF-8."
        ) from exc

    if len(vocabulary) != declared_count:
        raise EmbeddingFormatError(
            f"{embedding_path}: header declares {declared_count} vectors, "
            f"but {len(vocabulary)} were read."
        )

    matrix = np.stack(vectors).astype(requested_dtype, copy=False)
    return EmbeddingStore(vocabulary, matrix)
