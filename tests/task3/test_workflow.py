from __future__ import annotations

import csv
import json
from pathlib import Path

from src.task3.workflow import run_task3


def test_workflow_keeps_all_test_rows_and_marks_oov(tmp_path: Path) -> None:
    embeddings = tmp_path / "vectors.txt"
    embeddings.write_text(
        "4\n"
        "2\n"
        "nóng 1.0 0.0\n"
        "ấm 0.8 0.2\n"
        "lạnh -1.0 0.0\n"
        "rét -0.8 0.2\n",
        encoding="utf-8",
    )
    synonyms = tmp_path / "syn.txt"
    synonyms.write_text("nóng ấm\nlạnh rét\n", encoding="utf-8")
    antonyms = tmp_path / "ant.txt"
    antonyms.write_text("nóng lạnh\nấm rét\n", encoding="utf-8")

    test_dir = tmp_path / "vicon"
    test_dir.mkdir()
    header = "Word1\tWord2\tRelation\n"
    (test_dir / "400_noun_pairs.txt").write_text(
        header + "nóng\tấm\tSYN\n", encoding="utf-8"
    )
    (test_dir / "400_verb_pairs.txt").write_text(
        header + "nóng\tlạnh\tANT\n", encoding="utf-8"
    )
    (test_dir / "600_adj_pairs.txt").write_text(
        header + "nóng\tkhông_có\tANT\n", encoding="utf-8"
    )
    output = tmp_path / "results" / "predictions.csv"

    result = run_task3(
        embeddings_path=embeddings,
        synonym_path=synonyms,
        antonym_path=antonyms,
        test_dir=test_dir,
        output_path=output,
    )

    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 3
    assert [row["status"] for row in rows] == ["covered", "covered", "oov"]
    assert rows[2]["predicted"] == ""
    assert rows[2]["prob_syn"] == ""
    assert rows[2]["oov_words"] == "không_có"

    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["data"]["test"]["covered_pairs"] == 2
    assert metrics["evaluation"]["evaluated_rows"] == 2
    assert metrics["protocol"]["overlap_policy"].startswith("audit only")
    assert result.report_path.is_file()
