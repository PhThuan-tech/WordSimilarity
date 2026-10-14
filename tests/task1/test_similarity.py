import csv
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import pearsonr, spearmanr

from main import main
from src.common import EmbeddingStore, load_embeddings, load_visim
from src.common.datasets import ViSimPair
from src.task1.similarity import cosine_similarity, evaluate_visim


@pytest.mark.parametrize("a,b,expected", [
    ([1, 2], [1, 2], 1), ([1, 0], [0, 1], 0), ([1, 0], [-1, 0], -1),
    ([1e300, 0], [1e300, 0], 1), ([1e-300, 0], [1e-300, 0], 1),
])
def test_cosine_geometry_and_symmetry(a, b, expected):
    result = cosine_similarity(a, b)
    assert isinstance(result, float)
    assert result == pytest.approx(expected)
    assert result == pytest.approx(cosine_similarity(b, a))


@pytest.mark.parametrize("a,b", [
    ([], []), ([1], [1, 2]), ([[1]], [[1]]), ([0, 0], [1, 0]),
    ([np.nan], [1]), ([np.inf], [1]),
])
def test_invalid_vectors(a, b):
    with pytest.raises(ValueError):
        cosine_similarity(a, b)


def test_evaluation_filters_oov_zero_and_matches_scipy():
    store = EmbeddingStore(["sinh_viên", "a", "b", "c", "zero"],
                           np.array([[1, 0], [1, 0], [0, 1], [-1, 0], [0, 0]]))
    pairs = [ViSimPair("sinh_viên", w, "N", gold, 2 * gold, 0)
             for w, gold in [("a", 2), ("b", 3), ("c", 1), ("missing", 100), ("zero", 100)]]
    rows, summary = evaluate_visim(store, pairs)
    assert [r["status"] for r in rows] == ["covered"] * 3 + ["oov", "zero_norm"]
    assert rows[3]["oov_words"] == "missing"
    assert rows[3]["cosine"] is None and rows[4]["cosine"] is None
    assert summary["coverage"]["covered_pairs"] == 4
    assert summary["evaluated_pairs"] == 3
    assert summary["zero_norm_pairs"] == 1
    for key in ("sim1", "sim2"):
        gold = [r[key] for r in rows[:3]]
        assert summary["metrics"][key]["pearson"] == pytest.approx(pearsonr([1, 0, -1], gold).statistic)
        assert summary["metrics"][key]["spearman"] == pytest.approx(spearmanr([1, 0, -1], gold).statistic)


@pytest.mark.parametrize("count", [0, 1, 2])
def test_undefined_correlations_are_json_null(count):
    store = EmbeddingStore(["a"], np.array([[1, 0]]))
    rows, summary = evaluate_visim(store, [ViSimPair("a", "a", "N", 1, 2, 0)] * count)
    assert summary["metrics"]["sim1"]["pearson"] is None
    assert summary["metrics"]["sim1"]["spearman"] is None
    json.dumps(summary, allow_nan=False)


def test_cli_writes_unicode_csv_summary_and_is_repeatable(tmp_path, capsys):
    embeddings = tmp_path / "embeddings.txt"
    dataset = tmp_path / "visim.txt"
    embeddings.write_text("2\n2\nsinh_viên 1 0\na 0 1\n", encoding="utf-8")
    dataset.write_text("Word1\tWord2\tPOS\tSim1\tSim2\tSTD\n"
                       "sinh_viên\ta\tN\t1\t2\t0\n"
                       "missing\ta\tN\t2\t4\t0\n", encoding="utf-8")
    output = tmp_path / "nested" / "predictions.csv"
    args = ["similarity", "--embeddings", str(embeddings), "--dataset", str(dataset), "--output", str(output)]
    assert main(args) == 0
    original = output.read_bytes()
    with output.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and rows[0]["word1"] == "sinh_viên"
    assert rows[1]["cosine"] == "" and rows[1]["status"] == "oov"
    summary_file = output.with_name("predictions_summary.json")
    summary_original = summary_file.read_bytes()
    assert json.loads(summary_original)["evaluated_pairs"] == 1
    assert main(args) == 0
    assert output.read_bytes() == original
    assert summary_file.read_bytes() == summary_original
    assert "sim1" in capsys.readouterr().out
    with pytest.raises(ValueError, match="overwrite"):
        main(args[:-1] + [str(dataset)])


def test_real_visim_evaluation():
    root = Path(__file__).resolve().parents[2]
    store = load_embeddings(root / "data/word2vec/W2V_150.txt")
    rows, summary = evaluate_visim(store, load_visim(root / "data/ViSim-400/Visim-400.txt"))
    assert len(rows) == 400
    assert sum(r["cosine"] is not None for r in rows) == 344
    assert summary["coverage"]["oov_pairs"] == 56
    assert summary["coverage"]["pair_coverage"] == pytest.approx(0.86)
    for metric in summary["metrics"].values():
        assert metric["n"] == 344
        assert np.isfinite(metric["pearson"]) and np.isfinite(metric["spearman"])
