from __future__ import annotations

from pathlib import Path

import pytest

from src.common import (
    OutOfVocabularyError,
    get_coverage,
    load_embeddings,
    load_training_pairs,
    load_vicon,
    load_visim,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


@pytest.fixture(scope="module")
def embeddings():
    return load_embeddings(DATA / "word2vec" / "W2V_150.txt")


def test_real_embedding_shape_and_access(embeddings) -> None:
    assert embeddings.size == 77_021
    assert embeddings.dimension == 150
    assert embeddings.get_embedding_matrix().shape == (77_021, 150)
    assert embeddings.contains("sinh_viên")
    assert embeddings.get_vector("sinh_viên").shape == (150,)
    with pytest.raises(OutOfVocabularyError):
        embeddings.get_vector("__từ_chắc_chắn_không_tồn_tại__")


def test_real_visim_count_and_coverage(embeddings) -> None:
    rows = load_visim(DATA / "ViSim-400" / "Visim-400.txt")
    report = get_coverage(rows, embeddings)

    assert len(rows) == 400
    assert report.covered_pairs == 344
    assert report.oov_pairs == 56


def test_real_vicon_counts_labels_pos_and_coverage(embeddings) -> None:
    rows = load_vicon(DATA / "ViCon-400")
    report = get_coverage(rows, embeddings)

    assert len(rows) == 1_400
    assert sum(row.relation == "SYN" for row in rows) == 700
    assert sum(row.relation == "ANT" for row in rows) == 700
    assert sum(row.pos == "N" for row in rows) == 400
    assert sum(row.pos == "V" for row in rows) == 400
    assert sum(row.pos == "A" for row in rows) == 600
    assert report.covered_pairs == 1_173


def test_real_training_counts_and_malformed_rows() -> None:
    dataset = load_training_pairs(
        DATA / "antonym-synonym" / "Synonym_vietnamese.txt",
        DATA / "antonym-synonym" / "Antonym_vietnamese.txt",
    )

    assert dataset.diagnostics.total_rows == 13_562
    assert dataset.diagnostics.valid_rows == 13_558
    assert dataset.diagnostics.label_counts == {"SYN": 11_558, "ANT": 2_000}
    assert len(dataset.diagnostics.malformed_rows) == 4
    assert {row.line_number for row in dataset.diagnostics.malformed_rows} == {
        563,
        2_942,
        3_098,
        3_196,
    }
    assert len(dataset.diagnostics.conflicting_unordered_pairs) == 7
