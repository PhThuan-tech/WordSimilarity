"""Command-line entry point shared by the three assignment tasks."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EMBEDDINGS = PROJECT_ROOT / "data" / "word2vec" / "W2V_150.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Vietnamese word similarity assignment",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    similarity = subparsers.add_parser(
        "similarity", help="Task 1: evaluate cosine similarity on ViSim-400"
    )
    similarity.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    similarity.add_argument(
        "--dataset",
        type=Path,
        default=PROJECT_ROOT / "data" / "ViSim-400" / "Visim-400.txt",
    )
    similarity.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "task1_predictions.csv"
    )

    nearest = subparsers.add_parser(
        "nearest", help="Task 2: find k nearest words"
    )
    nearest.add_argument("--word", required=True, help="Vietnamese query token")
    nearest.add_argument("--k", type=int, default=10)
    nearest.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)

    classify = subparsers.add_parser(
        "classify", help="Task 3: train and evaluate a relation classifier"
    )
    classify.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS)
    classify.add_argument(
        "--synonyms",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "antonym-synonym"
        / "Synonym_vietnamese.txt",
    )
    classify.add_argument(
        "--antonyms",
        type=Path,
        default=PROJECT_ROOT
        / "data"
        / "antonym-synonym"
        / "Antonym_vietnamese.txt",
    )
    classify.add_argument(
        "--test-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "ViCon-400",
    )
    classify.add_argument(
        "--output", type=Path, default=PROJECT_ROOT / "results" / "task3_predictions.csv"
    )
    return parser


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "similarity":
        from src.task1.cli import run
    elif args.command == "nearest":
        from src.task2.cli import run
    else:
        from src.task3.cli import run
    return run(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "nearest" and args.k <= 0:
        build_parser().error("--k must be a positive integer")
    try:
        return _dispatch(args)
    except NotImplementedError as exc:
        print(exc)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
