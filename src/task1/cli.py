"""Task 1 command orchestration."""

from argparse import Namespace
import csv
import json
from pathlib import Path

from src.common import load_embeddings, load_visim
from .similarity import evaluate_visim


def run(args: Namespace) -> int:
    """Write predictions and a sibling JSON summary; return success."""
    output = Path(args.output)
    summary_path = output.with_name(output.stem + "_summary.json")
    if any(path.resolve() == source.resolve()
           for path in (output, summary_path)
           for source in (Path(args.embeddings), Path(args.dataset))):
        raise ValueError("Output must not overwrite an input file.")
    rows, summary = evaluate_visim(load_embeddings(args.embeddings), load_visim(args.dataset))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "word1", "word2", "pos", "sim1", "sim2", "std", "cosine", "status", "oov_words"
        ])
        writer.writeheader()
        writer.writerows(rows)
    serialized = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)
    summary_path.write_text(serialized + "\n", encoding="utf-8")
    # Escaped console JSON also works on Windows terminals using legacy encodings.
    print(json.dumps(summary, ensure_ascii=True, indent=2, allow_nan=False))
    print(f"Predictions: {output}")
    print(f"Summary: {summary_path}")
    return 0
