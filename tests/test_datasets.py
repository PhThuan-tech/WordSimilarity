from __future__ import annotations

from pathlib import Path

import pytest

from src.common.datasets import (
    DatasetFormatError,
    load_training_pairs,
    load_vicon,
    load_visim,
)


def test_load_visim_preserves_utf8_and_scores(tmp_path: Path) -> None:
    path = tmp_path / "visim.txt"
    path.write_text(
        "Word1\tWord2\tPOS\tSim1\tSim2\tSTD\n"
        "sinh_viên\thọc_sinh\tN\t4.2\t7.0\t0.5\n",
        encoding="utf-8",
    )

    rows = load_visim(path)

    assert len(rows) == 1
    assert rows[0].word1 == "sinh_viên"
    assert rows[0].sim1 == 4.2
    assert rows[0].sim2 == 7.0


def test_load_vicon_directory_has_deterministic_pos_order(tmp_path: Path) -> None:
    for filename, pair in (
        ("400_noun_pairs.txt", "nhà\tcửa\tSYN"),
        ("400_verb_pairs.txt", "vào\tra\tANT"),
        ("600_adj_pairs.txt", "đẹp\txấu\tANT"),
    ):
        (tmp_path / filename).write_text(
            f"Word1\tWord2\tRelation\n{pair}\n", encoding="utf-8"
        )

    rows = load_vicon(tmp_path)

    assert [row.pos for row in rows] == ["N", "V", "A"]
    assert [row.relation for row in rows] == ["SYN", "ANT", "ANT"]


def test_training_loader_reports_malformed_duplicates_and_conflicts(
    tmp_path: Path,
) -> None:
    synonyms = tmp_path / "syn.txt"
    antonyms = tmp_path / "ant.txt"
    synonyms.write_text(
        "xinh đẹp\nxinh đẹp\nđẹp xinh\nthiếu\n", encoding="utf-8"
    )
    antonyms.write_text("xinh đẹp\ncao thấp extra\n", encoding="utf-8")

    dataset = load_training_pairs(synonyms, antonyms)
    diagnostics = dataset.diagnostics

    assert len(dataset.pairs) == 4
    assert diagnostics.total_rows == 6
    assert diagnostics.valid_rows == 4
    assert diagnostics.label_counts == {"SYN": 3, "ANT": 1}
    assert len(diagnostics.malformed_rows) == 2
    assert diagnostics.duplicate_ordered_rows == 1
    assert diagnostics.duplicate_unordered_rows == 2
    assert diagnostics.conflicting_unordered_pairs == {
        ("xinh", "đẹp"): ("ANT", "SYN")
    }


def test_rejects_unknown_vicon_relation(tmp_path: Path) -> None:
    path = tmp_path / "pairs.txt"
    path.write_text(
        "Word1\tWord2\tRelation\na\tb\tRELATED\n", encoding="utf-8"
    )

    with pytest.raises(DatasetFormatError, match="unsupported relation"):
        load_vicon(path)
