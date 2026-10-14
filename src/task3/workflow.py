"""End-to-end Task 3 workflow and reproducible result writers."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.common import (
    CoverageReport,
    LabeledWordPair,
    TrainingDataset,
    get_coverage,
    load_embeddings,
    load_training_pairs,
    load_vicon,
)

from .classifier import (
    DEFAULT_RANDOM_STATE,
    audit_overlap,
    evaluate_predictions,
    synonym_probabilities,
    train_classifier,
)
from .features import FeatureBatch, build_feature_matrix


@dataclass(frozen=True, slots=True)
class Task3Result:
    predictions_path: Path
    metrics_path: Path
    report_path: Path
    metrics: dict[str, Any]


def _coverage_dict(report: CoverageReport) -> dict[str, Any]:
    return {
        "total_pairs": report.total_pairs,
        "covered_pairs": report.covered_pairs,
        "oov_pairs": report.oov_pairs,
        "pair_coverage": report.pair_coverage,
        "total_unique_words": report.total_unique_words,
        "covered_unique_words": report.covered_unique_words,
        "word_coverage": report.word_coverage,
        "oov_unique_words": list(report.oov_unique_words),
        "oov_occurrences": report.oov_occurrences,
    }


def _training_diagnostics(dataset: TrainingDataset) -> dict[str, Any]:
    diagnostics = dataset.diagnostics
    return {
        "total_rows": diagnostics.total_rows,
        "valid_rows": diagnostics.valid_rows,
        "label_counts": diagnostics.label_counts,
        "malformed_row_count": len(diagnostics.malformed_rows),
        "malformed_rows": [
            {
                "source": str(row.source),
                "line_number": row.line_number,
                "raw_value": row.raw_value,
                "reason": row.reason,
            }
            for row in diagnostics.malformed_rows
        ],
        "duplicate_ordered_rows": diagnostics.duplicate_ordered_rows,
        "duplicate_unordered_rows": diagnostics.duplicate_unordered_rows,
        "conflicting_unordered_pair_count": len(
            diagnostics.conflicting_unordered_pairs
        ),
        "conflicting_unordered_pairs": [
            {"word1": pair[0], "word2": pair[1], "labels": list(labels)}
            for pair, labels in sorted(
                diagnostics.conflicting_unordered_pairs.items()
            )
        ],
    }


def _prediction_rows(
    pairs: list[LabeledWordPair],
    batch: FeatureBatch,
    predicted: np.ndarray,
    probabilities: np.ndarray,
) -> list[dict[str, object]]:
    prediction_by_index = {
        source_index: (str(label), float(probability))
        for source_index, label, probability in zip(
            batch.source_indices, predicted, probabilities, strict=True
        )
    }
    rows: list[dict[str, object]] = []
    for index, pair in enumerate(pairs):
        prediction = prediction_by_index.get(index)
        if prediction is None:
            rows.append(
                {
                    "word1": pair.word1,
                    "word2": pair.word2,
                    "pos": pair.pos or "",
                    "gold": pair.relation,
                    "predicted": "",
                    "prob_syn": "",
                    "status": "oov",
                    "oov_words": ";".join(batch.oov_by_index[index]),
                }
            )
        else:
            label, probability = prediction
            rows.append(
                {
                    "word1": pair.word1,
                    "word2": pair.word2,
                    "pos": pair.pos or "",
                    "gold": pair.relation,
                    "predicted": label,
                    "prob_syn": f"{probability:.8f}",
                    "status": "covered",
                    "oov_words": "",
                }
            )
    return rows


def _write_predictions(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "word1",
        "word2",
        "pos",
        "gold",
        "predicted",
        "prob_syn",
        "status",
        "oov_words",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_metrics(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    evaluation = metrics["evaluation"]
    data = metrics["data"]
    overlap = metrics["overlap_audit"]
    training = metrics["training_diagnostics"]
    per_class = evaluation["per_class"]
    confusion = evaluation["confusion_matrix"]

    lines = [
        "# Task 3 — Synonym–Antonym Classification",
        "",
        "## Protocol",
        "",
        "- Features: `[abs(v1-v2), v1*v2, cosine(v1,v2)]`.",
        "- Model: StandardScaler + class-balanced Logistic Regression.",
        f"- Random seed: `{metrics['protocol']['random_state']}`.",
        "- Duplicate and conflicting valid training rows are retained.",
        "- OOV rows remain in the CSV but are excluded from model metrics.",
        "- The overlap audit is reporting-only; predictions never use label lookup.",
        "",
        "## Data and coverage",
        "",
        "| Split | Total | Covered | OOV | Coverage |",
        "|---|---:|---:|---:|---:|",
        (
            f"| Train | {data['train']['total_pairs']} | "
            f"{data['train']['covered_pairs']} | {data['train']['oov_pairs']} | "
            f"{data['train']['pair_coverage']:.2%} |"
        ),
        (
            f"| ViCon test | {data['test']['total_pairs']} | "
            f"{data['test']['covered_pairs']} | {data['test']['oov_pairs']} | "
            f"{data['test']['pair_coverage']:.2%} |"
        ),
        "",
        "## Evaluation on covered ViCon rows",
        "",
        f"Evaluated rows: **{evaluation['evaluated_rows']}**",
        "",
        "| Label | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in ("ANT", "SYN"):
        row = per_class[label]
        lines.append(
            f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['f1']:.4f} | {row['support']} |"
        )
    lines.extend(
        [
            (
                f"| Macro avg | {evaluation['macro_avg']['precision']:.4f} | "
                f"{evaluation['macro_avg']['recall']:.4f} | "
                f"{evaluation['macro_avg']['f1']:.4f} | "
                f"{evaluation['macro_avg']['support']} |"
            ),
            "",
            f"Accuracy: **{evaluation['accuracy']:.4f}**",
            "",
            "Confusion matrix (rows = gold, columns = prediction; order ANT, SYN):",
            "",
            "```text",
            f"{confusion[0]}",
            f"{confusion[1]}",
            "```",
            "",
            "## Data-quality and overlap audit",
            "",
            f"- Malformed training rows: {training['malformed_row_count']}.",
            f"- Ordered duplicate rows retained: {training['duplicate_ordered_rows']}.",
            f"- Unordered duplicate rows retained: {training['duplicate_unordered_rows']}.",
            (
                "- Conflicting unordered training pairs retained: "
                f"{training['conflicting_unordered_pair_count']}."
            ),
            f"- Overlapping unique train/test pairs: {overlap['overlapping_unique_pairs']}.",
            f"- Overlapping ViCon rows: {overlap['overlapping_test_rows']}.",
            (
                "- ViCon rows whose train pair has ambiguous labels: "
                f"{overlap['ambiguous_train_label_rows']}."
            ),
            "",
            "## Interpretation limit",
            "",
            (
                "ViCon substantially overlaps the supplied training pairs when word order "
                "is ignored. These required-protocol metrics therefore do not measure "
                "clean generalization to unseen pairs and must not be presented as such."
            ),
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_task3(
    *,
    embeddings_path: str | Path,
    synonym_path: str | Path,
    antonym_path: str | Path,
    test_dir: str | Path,
    output_path: str | Path,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> Task3Result:
    """Train, evaluate, and write all required Task 3 artifacts."""

    embeddings = load_embeddings(embeddings_path)
    training = load_training_pairs(synonym_path, antonym_path)
    test_pairs = load_vicon(test_dir)

    train_pairs = list(training.pairs)
    train_coverage = get_coverage(train_pairs, embeddings)
    test_coverage = get_coverage(test_pairs, embeddings)
    overlap = audit_overlap(train_pairs, test_pairs)

    train_batch = build_feature_matrix(train_pairs, embeddings)
    test_batch = build_feature_matrix(test_pairs, embeddings)
    model = train_classifier(
        train_batch.matrix,
        train_batch.labels,
        random_state=random_state,
    )
    predicted = model.predict(test_batch.matrix)
    probabilities = synonym_probabilities(model, test_batch.matrix)
    evaluation = evaluate_predictions(test_batch.labels, predicted)

    output = Path(output_path)
    metrics_path = output.with_name(f"{output.stem}_metrics.json")
    report_path = output.with_name(f"{output.stem}_report.md")
    prediction_rows = _prediction_rows(
        test_pairs, test_batch, predicted, probabilities
    )

    metrics: dict[str, Any] = {
        "protocol": {
            "task": "synonym_antonym_classification",
            "features": ["absolute_difference", "elementwise_product", "cosine"],
            "feature_dimension": int(train_batch.matrix.shape[1]),
            "model": "StandardScaler + LogisticRegression",
            "class_weight": "balanced",
            "random_state": random_state,
            "positive_class": "SYN",
            "duplicate_policy": "retain valid rows",
            "conflict_policy": "retain and report valid rows",
            "oov_policy": "retain in CSV; exclude from training and metrics",
            "overlap_policy": "audit only; never use train label lookup",
        },
        "data": {
            "embedding_vocabulary_size": embeddings.size,
            "embedding_dimension": embeddings.dimension,
            "train": _coverage_dict(train_coverage),
            "test": _coverage_dict(test_coverage),
            "train_feature_rows": train_batch.covered_rows,
            "test_feature_rows": test_batch.covered_rows,
        },
        "training_diagnostics": _training_diagnostics(training),
        "overlap_audit": overlap.to_dict(),
        "evaluation": evaluation,
    }

    _write_predictions(output, prediction_rows)
    _write_metrics(metrics_path, metrics)
    _write_report(report_path, metrics)
    return Task3Result(
        predictions_path=output,
        metrics_path=metrics_path,
        report_path=report_path,
        metrics=metrics,
    )
