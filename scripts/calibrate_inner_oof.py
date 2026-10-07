#!/usr/bin/env python3
"""Fit temperature on run-grouped OOF scores and assess calibration/stopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.screen_shrinkage_lda import _character_metrics, _decode_strings  # noqa: E402
from src.evaluation.calibration import calibration_metrics  # noqa: E402
from src.models.classifier_interface import SklearnP300Adapter  # noqa: E402


def run(scores_path: Path, output_path: Path, model_name: str | None = None) -> dict:
    with np.load(scores_path, allow_pickle=False) as stored:
        if "logits" in stored:
            logits = stored["logits"].astype(np.float64)
        else:
            names = _decode_strings(stored["model_names"])
            selected = model_name or names[0]
            logits = stored["logits_by_model"][list(names).index(selected)].astype(np.float64)
        y = stored["y"].astype(np.uint8)
        run_id = _decode_strings(stored["run_id"])
        subject = _decode_strings(stored["subject"])
        char_idx = stored["char_idx"]
        seq_idx = stored["seq_idx"]
        stim_code = stored["stim_code"]

    valid = np.isfinite(logits)
    model = SklearnP300Adapter(estimator=None)
    calibrated = np.full_like(logits, np.nan)
    calibrated[valid] = model.calibrate(logits[valid], y[valid])
    result = {
        "scores": str(scores_path),
        "calibration_fit": "all run-grouped out-of-fold training-pool logits",
        "temperature": model.temperature,
        "epochs": int(valid.sum()),
        "subjects": int(len(set(subject[valid]))),
        "before": calibration_metrics(logits[valid], y[valid]),
        "after": calibration_metrics(calibrated[valid], y[valid]),
        "before_characters": _character_metrics(
            logits[valid], y[valid], run_id[valid], char_idx[valid],
            seq_idx[valid], stim_code[valid],
        ),
        "after_characters": _character_metrics(
            calibrated[valid], y[valid], run_id[valid], char_idx[valid],
            seq_idx[valid], stim_code[valid],
        ),
        "per_subject_temperature": "not fit; only 17 subject groups, inspect data sufficiency first",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-name")
    args = parser.parse_args()
    result = run(args.scores, args.output, args.model_name)
    print(json.dumps({
        "temperature": result["temperature"],
        "nll_before_after": [result["before"]["nll"], result["after"]["nll"]],
        "ece_before_after": [result["before"]["ece"], result["after"]["ece"]],
        "stopping_acc_before_after": [
            result["before_characters"]["locked_stopping_accuracy"],
            result["after_characters"]["locked_stopping_accuracy"],
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
