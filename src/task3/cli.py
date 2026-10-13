"""Command-line integration for Task 3."""

from argparse import Namespace

from .workflow import run_task3


def run(args: Namespace) -> int:
    """Train/evaluate the baseline and print a concise result summary."""

    result = run_task3(
        embeddings_path=args.embeddings,
        synonym_path=args.synonyms,
        antonym_path=args.antonyms,
        test_dir=args.test_dir,
        output_path=args.output,
    )
    evaluation = result.metrics["evaluation"]
    train = result.metrics["data"]["train"]
    test = result.metrics["data"]["test"]
    overlap = result.metrics["overlap_audit"]

    print("Task 3 completed")
    print(
        f"Train coverage: {train['covered_pairs']}/{train['total_pairs']} "
        f"({train['pair_coverage']:.2%})"
    )
    print(
        f"ViCon coverage: {test['covered_pairs']}/{test['total_pairs']} "
        f"({test['pair_coverage']:.2%})"
    )
    print(
        "Train/test overlap (unordered unique pairs): "
        f"{overlap['overlapping_unique_pairs']}"
    )
    print(f"Accuracy: {evaluation['accuracy']:.4f}")
    print(f"Macro-F1: {evaluation['macro_avg']['f1']:.4f}")
    print(f"Predictions: {result.predictions_path}")
    print(f"Metrics: {result.metrics_path}")
    print(f"Report: {result.report_path}")
    return 0
