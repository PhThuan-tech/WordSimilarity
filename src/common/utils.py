"""Small shared utilities that contain no task-specific algorithms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Protocol

from .embeddings import EmbeddingStore


class WordPair(Protocol):
    word1: str
    word2: str


@dataclass(frozen=True, slots=True)
class CoverageReport:
    total_pairs: int
    covered_pairs: int
    oov_pairs: int
    total_unique_words: int
    covered_unique_words: int
    oov_unique_words: tuple[str, ...]
    oov_occurrences: dict[str, int]

    @property
    def pair_coverage(self) -> float:
        """Fraction of pairs for which both words have vectors."""

        return self.covered_pairs / self.total_pairs if self.total_pairs else 0.0

    @property
    def word_coverage(self) -> float:
        """Fraction of unique words found in the embedding vocabulary."""

        return (
            self.covered_unique_words / self.total_unique_words
            if self.total_unique_words
            else 0.0
        )


def get_coverage(
    pairs: Iterable[WordPair],
    embeddings: EmbeddingStore,
) -> CoverageReport:
    """Compute pair and vocabulary coverage without changing or dropping rows."""

    materialized = list(pairs)
    all_words = {word for pair in materialized for word in (pair.word1, pair.word2)}
    oov_counter: Counter[str] = Counter()
    covered_pairs = 0

    for pair in materialized:
        missing = [
            word
            for word in (pair.word1, pair.word2)
            if not embeddings.contains(word)
        ]
        if missing:
            oov_counter.update(missing)
        else:
            covered_pairs += 1

    oov_words = tuple(sorted(word for word in all_words if word not in embeddings))
    return CoverageReport(
        total_pairs=len(materialized),
        covered_pairs=covered_pairs,
        oov_pairs=len(materialized) - covered_pairs,
        total_unique_words=len(all_words),
        covered_unique_words=len(all_words) - len(oov_words),
        oov_unique_words=oov_words,
        oov_occurrences=dict(sorted(oov_counter.items())),
    )
