"""CLI integration point for Member 2 (Task 2: K-nearest Words)."""

import sys
from argparse import Namespace
from src.common import load_embeddings, OutOfVocabularyError
from src.task2.nearest import NearestWordSearcher


def run(args: Namespace) -> int:
    """Orchestrates Task 2: Find top-k nearest words using vectorized cosine similarity."""
    word = args.word
    k = args.k

    print("Đang tải dữ liệu Word2Vec...")
    embeddings = load_embeddings("data/word2vec/W2V_150.txt")

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