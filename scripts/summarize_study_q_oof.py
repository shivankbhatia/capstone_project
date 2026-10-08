#!/usr/bin/env python3
"""Summarize an existing clean Study Q OOF sidecar by condition."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import h5py
import numpy as np
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_study_q_classifier_cv import _character_metrics  # noqa: E402


def summarize(scores_path: Path, cache_path: Path, output_path: Path) -> dict:
    sidecar = np.load(scores_path, allow_pickle=False)
    logits = sidecar["logits"]
    y = sidecar["y"].astype(np.uint8)
    run_id = sidecar["run_id"].astype(str)
    subject = sidecar["subject"].astype(str)
    session = sidecar["session"].astype(str)
    char_idx = sidecar["char_idx"].astype(np.int32)
    seq_idx = sidecar["seq_idx"].astype(np.int32)
    lit_mask = sidecar["lit_mask"].astype(np.uint8)
    with h5py.File(cache_path, "r") as h5:
        cache_runs = h5["run_id"][:].astype(str)
        cache_conditions = h5["condition"][:].astype(str)
    if len(cache_runs) != len(run_id) or not np.array_equal(cache_runs, run_id):
        raise ValueError("OOF sidecar and training cache row order differ")
    valid = np.isfinite(logits)
    records = {}
    for condition in sorted(set(cache_conditions)):
        mask = valid & (cache_conditions == condition)
        subset = {
            "y": y[mask], "run_id": run_id[mask], "subject": subject[mask],
            "session": session[mask], "char_idx": char_idx[mask],
            "seq_idx": seq_idx[mask], "lit_mask": lit_mask[mask],
        }
        records[condition] = {
            "epochs": int(mask.sum()),
            "epoch_auc": float(roc_auc_score(y[mask], logits[mask])),
            "balanced_accuracy_at_zero": float(balanced_accuracy_score(y[mask], logits[mask] > 0)),
            "character_metrics": _character_metrics(logits[mask], subset),
        }
    result = {
        "study": "StudyQ", "source_scores": str(scores_path),
        "source_cache": str(cache_path),
        "selection_scope": "clean Train runs only; session-grouped OOF; held-out session SE003 untouched",
        "per_condition": records,
    }
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, default=ROOT / "results/tables/study_q_classifier_inner_scores.npz")
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/study_q_classifier_epochs.h5")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/study_q_classifier_by_condition.json")
    args = parser.parse_args()
    result = summarize(args.scores, args.cache, args.output)
    print(json.dumps({k: {"epoch_auc": v["epoch_auc"], "character_accuracy": v["character_metrics"]["full_available_sequence_accuracy"], "characters": v["character_metrics"]["characters"]} for k, v in result["per_condition"].items()}, indent=2))
