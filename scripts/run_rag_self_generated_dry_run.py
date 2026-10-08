#!/usr/bin/env python3
"""End-to-end software dry run on explicitly synthetic self-generated text."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.prepare_participant_text_sample import prepare
from src.models.llm_predictor import LLMPredictor
from src.models.rag_predictor import RAGPredictor


def run() -> dict:
    # Deliberately synthetic repeated prose; it is only a data-flow test.
    paragraph = (
        "A quiet learner reads a clear page, writes a short note, and checks each word. "
        "The local spelling assistant keeps useful phrases nearby and updates after text is decoded. "
    )
    sample = paragraph * 70
    local = ROOT / "data/rag/participant_local/self_generated_dry_run"
    local.mkdir(parents=True, exist_ok=True)
    source = local / "synthetic_sample.txt"
    source.write_text(sample, encoding="utf-8")
    counts = prepare(source, local)
    splits = json.loads((local / "splits.json").read_text(encoding="utf-8"))

    layout = json.loads((ROOT / "data/processed/grid_layout.json").read_text())[
        "StudyD"
    ]
    grid = np.empty((layout["n_rows"], layout["n_cols"]), dtype=object)
    for label, (row, col) in layout["grid_map"].items():
        grid[row - 1, col - 1] = label
    base = LLMPredictor(grid, local_files_only=True)
    model = RAGPredictor(
        base,
        phrase_bank_path=local / "phrase_bank_global.csv",
        rag_weight=0.25,
    )
    prefix = splits["bank_training_text"] + " " + splits["gate_tuning_text"]
    test_text = splits["evaluation_text"]
    rows = []
    for offset, char in enumerate(test_text):
        if char.isalpha() and len(rows) < 12:
            context = (prefix + " " + test_text[:offset])[-128:]
            prior = model.predict_next_char(context)
            rows.append({"target": char.lower(), "finite": bool(np.isfinite(prior).all()),
                         "normalized": bool(np.isclose(prior.sum(), 1.0))})
    return {
        "status": "software_dry_run_only",
        "text_source": "assistant-generated synthetic repeated prose",
        "participant_data": False,
        "counts": counts,
        "scored_characters": len(rows),
        "all_priors_finite_and_normalized": all(r["finite"] and r["normalized"] for r in rows),
        "evaluation_tail_added_to_bank": False,
    }


if __name__ == "__main__":
    result = run()
    out = ROOT / "results/tables/rag_self_generated_dry_run.json"
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
