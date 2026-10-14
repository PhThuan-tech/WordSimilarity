"""CLI integration point for Member 2 (Task 2: K-nearest Words)."""

import sys
from argparse import Namespace
from src.common import load_embeddings, OutOfVocabularyError
from src.task2.nearest import NearestWordSearcher


def _configure_utf8_stdout() -> None:
    """Make Vietnamese output safe on Windows consoles using legacy encodings."""

    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfigure):
        reconfigure(encoding="utf-8", errors="backslashreplace")


def run(args: Namespace) -> int:
    """Orchestrates Task 2: Find top-k nearest words using vectorized cosine similarity."""
    _configure_utf8_stdout()
    word = args.word
    k = args.k

    print("Đang tải dữ liệu Word2Vec...")
    embeddings = load_embeddings(args.embeddings)

    try:
        searcher = NearestWordSearcher(embeddings)
        neighbors = searcher.find_nearest(word, k)

        print(f"\nTop {len(neighbors)} từ gần nhất với '{word}':")
        print("-" * 45)
        print(f"{'Rank':<6}{'Token':<25}{'Cosine Sim':<12}")
        print("-" * 45)

        for rank, item in enumerate(neighbors, 1):
            print(f"{rank:<6}{item.word:<25}{item.similarity:.6f}")

        return 0

    except OutOfVocabularyError as e:
        print(f"[LỖI OOV] {e}")
        return 1
    except ValueError as e:
        print(f"[LỖI THAM SỐ] {e}")
        return 1
