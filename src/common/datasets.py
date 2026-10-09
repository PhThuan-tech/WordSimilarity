"""Validated readers for ViSim, ViCon, and synonym/antonym training data."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Relation = Literal["SYN", "ANT"]


class DatasetFormatError(ValueError):
    """Raised when an evaluation dataset has an invalid schema or value."""


@dataclass(frozen=True, slots=True)
class ViSimPair:
    word1: str
    word2: str
    pos: str
    sim1: float
    sim2: float
    std: float


@dataclass(frozen=True, slots=True)
class LabeledWordPair:
    word1: str
    word2: str
    relation: Relation
    pos: str | None = None


@dataclass(frozen=True, slots=True)
class MalformedRow:
    source: Path
    line_number: int
    raw_value: str
    reason: str


@dataclass(frozen=True, slots=True)
class TrainingDiagnostics:
    total_rows: int
    valid_rows: int
    label_counts: dict[str, int]
    malformed_rows: tuple[MalformedRow, ...]
    duplicate_ordered_rows: int
    duplicate_unordered_rows: int
    conflicting_unordered_pairs: dict[tuple[str, str], tuple[str, ...]]


@dataclass(frozen=True, slots=True)
class TrainingDataset:
    """Training rows and diagnostics; valid duplicate rows remain untouched."""

    pairs: tuple[LabeledWordPair, ...]
    diagnostics: TrainingDiagnostics


_VISIM_HEADER = ("Word1", "Word2", "POS", "Sim1", "Sim2", "STD")
_VICON_HEADER = ("Word1", "Word2", "Relation")
_VICON_FILES: tuple[tuple[str, str], ...] = (
    ("400_noun_pairs.txt", "N"),
    ("400_verb_pairs.txt", "V"),
    ("600_adj_pairs.txt", "A"),
)


def _require_file(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {resolved}")
    return resolved


def _read_tsv(path: Path) -> tuple[tuple[str, ...], list[tuple[int, list[str]]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle, delimiter="\t")
            try:
                header = tuple(next(reader))
            except StopIteration as exc:
                raise DatasetFormatError(f"{path}: dataset is empty.") from exc
            rows = [(line_number, row) for line_number, row in enumerate(reader, 2)]
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(f"{path}: file is not valid UTF-8.") from exc
    return header, rows


def _parse_finite_float(value: str, path: Path, line_number: int, column: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise DatasetFormatError(
            f"{path}: line {line_number}, {column} must be numeric."
        ) from exc
    if number != number or number in (float("inf"), float("-inf")):
        raise DatasetFormatError(
            f"{path}: line {line_number}, {column} must be finite."
        )
    return number


def load_visim(path: str | Path) -> list[ViSimPair]:
    """Load ViSim-400 without modifying words or human scores."""

    dataset_path = _require_file(path)
    header, rows = _read_tsv(dataset_path)
    if header != _VISIM_HEADER:
        raise DatasetFormatError(
            f"{dataset_path}: expected header {_VISIM_HEADER}, got {header}."
        )

    result: list[ViSimPair] = []
    for line_number, row in rows:
        if len(row) != 6:
            raise DatasetFormatError(
                f"{dataset_path}: line {line_number} has {len(row)} columns; expected 6."
            )
        word1, word2, pos, sim1, sim2, std = (value.strip() for value in row)
        if not word1 or not word2 or not pos:
            raise DatasetFormatError(
                f"{dataset_path}: line {line_number} has an empty required field."
            )
        result.append(
            ViSimPair(
                word1=word1,
                word2=word2,
                pos=pos,
                sim1=_parse_finite_float(sim1, dataset_path, line_number, "Sim1"),
                sim2=_parse_finite_float(sim2, dataset_path, line_number, "Sim2"),
                std=_parse_finite_float(std, dataset_path, line_number, "STD"),
            )
        )
    return result


def _load_vicon_file(path: Path, pos: str | None) -> list[LabeledWordPair]:
    header, rows = _read_tsv(path)
    if header != _VICON_HEADER:
        raise DatasetFormatError(
            f"{path}: expected header {_VICON_HEADER}, got {header}."
        )

    result: list[LabeledWordPair] = []
    for line_number, row in rows:
        if len(row) != 3:
            raise DatasetFormatError(
                f"{path}: line {line_number} has {len(row)} columns; expected 3."
            )
        word1, word2, relation = (value.strip() for value in row)
        if not word1 or not word2:
            raise DatasetFormatError(f"{path}: line {line_number} has an empty word.")
        if relation not in ("SYN", "ANT"):
            raise DatasetFormatError(
                f"{path}: line {line_number} has unsupported relation {relation!r}."
            )
        result.append(
            LabeledWordPair(
                word1=word1,
                word2=word2,
                relation=relation,
                pos=pos,
            )
        )
    return result


def load_vicon(path: str | Path) -> list[LabeledWordPair]:
    """Load one ViCon file or all three standard files from a directory.

    When a directory is supplied, output order is noun, verb, adjective. The
    ``pos`` value is inferred as ``N``, ``V`` or ``A`` from the standard file.
    For an arbitrary single file, ``pos`` is ``None``.
    """

    dataset_path = Path(path)
    if dataset_path.is_file():
        inferred_pos = next(
            (pos for filename, pos in _VICON_FILES if dataset_path.name == filename),
            None,
        )
        return _load_vicon_file(dataset_path, inferred_pos)
    if dataset_path.is_dir():
        result: list[LabeledWordPair] = []
        for filename, pos in _VICON_FILES:
            file_path = dataset_path / filename
            if not file_path.is_file():
                raise FileNotFoundError(f"Required ViCon file does not exist: {file_path}")
            result.extend(_load_vicon_file(file_path, pos))
        return result
    raise FileNotFoundError(f"ViCon path does not exist: {dataset_path}")


def _read_training_file(
    path: Path,
    relation: Relation,
) -> tuple[list[LabeledWordPair], list[MalformedRow], int]:
    pairs: list[LabeledWordPair] = []
    malformed: list[MalformedRow] = []
    total = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline=None) as handle:
            for line_number, line in enumerate(handle, 1):
                total += 1
                fields = line.strip().split()
                if len(fields) != 2:
                    malformed.append(
                        MalformedRow(
                            source=path,
                            line_number=line_number,
                            raw_value=line.rstrip("\r\n"),
                            reason=f"expected 2 whitespace-separated words, got {len(fields)}",
                        )
                    )
                    continue
                pairs.append(
                    LabeledWordPair(
                        word1=fields[0],
                        word2=fields[1],
                        relation=relation,
                    )
                )
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(f"{path}: file is not valid UTF-8.") from exc
    return pairs, malformed, total


def load_training_pairs(
    synonym_path: str | Path,
    antonym_path: str | Path,
) -> TrainingDataset:
    """Load labeled training files and report, but do not remove, duplicates.

    Structurally invalid rows cannot form word pairs, so they are excluded from
    ``pairs`` and preserved verbatim in ``diagnostics.malformed_rows``.
    """

    synonym_file = _require_file(synonym_path)
    antonym_file = _require_file(antonym_path)
    synonyms, synonym_errors, synonym_total = _read_training_file(
        synonym_file, "SYN"
    )
    antonyms, antonym_errors, antonym_total = _read_training_file(
        antonym_file, "ANT"
    )
    pairs = synonyms + antonyms

    ordered_counts = Counter((p.word1, p.word2, p.relation) for p in pairs)
    unordered_counts = Counter(
        (tuple(sorted((p.word1, p.word2))), p.relation) for p in pairs
    )
    relations_by_pair: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    for pair in pairs:
        relations_by_pair[tuple(sorted((pair.word1, pair.word2)))].add(pair.relation)

    conflicts = {
        pair: tuple(sorted(relations))
        for pair, relations in relations_by_pair.items()
        if len(relations) > 1
    }
    diagnostics = TrainingDiagnostics(
        total_rows=synonym_total + antonym_total,
        valid_rows=len(pairs),
        label_counts=dict(Counter(p.relation for p in pairs)),
        malformed_rows=tuple(synonym_errors + antonym_errors),
        duplicate_ordered_rows=sum(count - 1 for count in ordered_counts.values()),
        duplicate_unordered_rows=sum(count - 1 for count in unordered_counts.values()),
        conflicting_unordered_pairs=conflicts,
    )
    return TrainingDataset(pairs=tuple(pairs), diagnostics=diagnostics)
