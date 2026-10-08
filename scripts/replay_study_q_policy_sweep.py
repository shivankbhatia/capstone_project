#!/usr/bin/env python3
"""Tune Q stopping on nested training-only OOF scores, without refitting M0."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.nested_study_q_calibration_fusion import (
    ROOT, _eval_policy, _llm_priors, _true_cells,
)


def _compact(metrics):
    output = {}
    for mode, values in metrics.items():
        row = {key: values[key] for key in ("correct", "n", "accuracy", "mean_sequences")}
        row["depth_histogram"] = {
            str(depth): count for depth, count in sorted(Counter(values["depth"]).items())
        }
        output[mode] = row
    return output


def run(scores_path: Path, result_path: Path) -> dict:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    with np.load(scores_path, allow_pickle=False) as scores:
        logits = scores["logits"]
        meta = {key: scores[key] for key in
                ("run_id", "subject", "session", "char_idx", "seq_idx", "y", "lit_mask")}
        rows = np.flatnonzero(np.isfinite(logits))
        true_cells = _true_cells(meta, rows)
        layout = json.loads((ROOT / "data/processed/grid_layout.json").read_text())["StudyQ"]
        grid = np.empty((layout["n_rows"], layout["n_cols"]), dtype=object)
        for symbol, (row, col) in layout["grid_map"].items():
            grid[row - 1, col - 1] = symbol
        priors = _llm_priors(meta, rows, true_cells, grid)

    default, _ = _eval_policy(logits, meta, rows, priors, tau=0.8,
                              min_sequences=2, max_sequences=10)
    default_acc = default["lm_fusion"]["accuracy"]
    candidates = []
    for min_count in (2, 3):
        for max_count in (5, 8, 10):
            if min_count > max_count:
                continue
            for threshold in (0.60, 0.70, 0.80, 0.90):
                metrics, _ = _eval_policy(
                    logits, meta, rows, priors, tau=threshold,
                    min_sequences=min_count, max_sequences=max_count,
                )
                candidates.append({"tau": threshold, "min_sequences": min_count,
                                   "max_sequences": max_count, "metrics": _compact(metrics)})
    eligible = [c for c in candidates
                if c["metrics"]["lm_fusion"]["accuracy"] >= default_acc]
    selected = min(eligible, key=lambda c: (
        c["metrics"]["lm_fusion"]["mean_sequences"],
        -c["metrics"]["lm_fusion"]["accuracy"],
    )) if eligible else None
    result["policy_tuning"] = {
        "data": "nested outer OOF predictions from clean Q training sessions only",
        "selection": "fewest mean sequences at no pooled fusion accuracy loss vs tau=.8/min=2/max=10; ties prefer higher accuracy",
        "default_reference": {"tau": 0.8, "min_sequences": 2, "max_sequences": 10,
                              "fusion_accuracy": default_acc,
                              "mean_sequences": default["lm_fusion"]["mean_sequences"]},
        "selected": selected,
        "sweep": candidates,
        "heldout_test_data_read": False,
    }
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result["policy_tuning"]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "results/tables/study_q_nested_fusion_scores.npz")
    parser.add_argument("--result", type=Path, default=ROOT / "results/tables/study_q_nested_fusion.json")
    args = parser.parse_args()
    print(json.dumps(run(args.scores, args.result), indent=2))
