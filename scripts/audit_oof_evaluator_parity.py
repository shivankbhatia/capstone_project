#!/usr/bin/env python3
"""Compare manifest-style and inner-fusion decoders on identical clean D OOF data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_pipeline import load_spelling_matrix  # noqa: E402
from scripts.evaluate_classifier_manifest import (  # noqa: E402
    _decode_oof_sequences, _sequence_rows,
)
from scripts.evaluate_inner_fusion import _decode_model, _summarize  # noqa: E402
from scripts.nested_sequence_calibration import _load_metadata  # noqa: E402
from scripts.screen_shrinkage_lda import _decode_strings  # noqa: E402
from src.models.llm_predictor import LLMPredictor  # noqa: E402


def run(scores_path: Path, cache_path: Path, output_path: Path,
        alphas=(0.0, 0.1, 0.85)):
    matrix, _, _ = load_spelling_matrix("data/processed/grid_layout.json", study_name="StudyD")
    llm = LLMPredictor(matrix, local_files_only=True)
    char_list = llm.char_list
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
    meta["run_id"] = _decode_strings(meta["run_id"])
    with np.load(scores_path, allow_pickle=False) as stored:
        logits = stored["M0_calibrated"].astype(float)

    target_codes = {}
    for rid, char_idx, label, code in zip(
        meta["run_id"], meta["char_idx"], meta["y"], meta["stim_code"]
    ):
        if int(char_idx) < 0 or not int(label):
            continue
        counts = target_codes.setdefault((str(rid), int(char_idx)), {})
        counts[int(code)] = counts.get(int(code), 0) + 1
    target_by_char = {}
    labels_by_run = {}
    for key, counts in target_codes.items():
        rows = [code for code in counts if 1 <= code <= 9]
        cols = [code for code in counts if 10 <= code <= 17]
        if rows and cols:
            row = max(rows, key=lambda code: counts[code]) - 1
            col = max(cols, key=lambda code: counts[code]) - 10
            target_by_char[key] = row * 8 + col
            labels_by_run.setdefault(key[0], {})[key[1]] = char_list[row * 8 + col]
    llm_priors = {}
    for rid, labels in labels_by_run.items():
        ordered = sorted(labels)
        for pos, char_idx in enumerate(ordered):
            context = "".join(labels[index] for index in ordered[:pos])
            llm_priors[(rid, char_idx)] = llm.predict_next_char(context)

    all_rows = {}
    by_char, _ = _sequence_rows(
        logits, meta["y"], meta["stim_code"], meta["char_idx"], meta["seq_idx"],
        char_list, run_ids=meta["run_id"],
    )
    parity_by_alpha = {}
    for alpha in alphas:
        inner, nll, calibration = _decode_model(
            logits, meta, char_list, llm_priors, tau=0.80, min_seq=2,
            max_seq=10, alpha=alpha,
        )
        manifest_style = _decode_oof_sequences(
            by_char, target_by_char, llm_priors, char_list, alpha=alpha,
            tau=0.80, min_seq=2, max_seq=10,
        )
        key = lambda row: (row["run_id"], row["char_idx"])
        left, right = {key(row): row for row in inner}, {key(row): row for row in manifest_style}
        shared = sorted(set(left) & set(right))
        fields = ("target_idx", "predicted_idx", "stopped_correct", "full_correct", "sequences_used")
        mismatches = [
            {"run_id": rid, "char_idx": idx,
             "fields": [field for field in fields if left[(rid, idx)][field] != right[(rid, idx)][field]]}
            for rid, idx in shared
            if any(left[(rid, idx)][field] != right[(rid, idx)][field] for field in fields)
        ]
        parity_by_alpha[str(alpha)] = {
            "inner_evaluator": _summarize(inner, nll, calibration),
            "manifest_style_evaluator": _summarize(manifest_style, nll, calibration),
            "characters_left": len(inner), "characters_right": len(manifest_style),
            "characters_compared": len(shared), "mismatch_count": len(mismatches),
            "mismatch_examples": mismatches[:10],
        }
    oof_char_keys = sorted(set((str(r), int(c)) for r, c in zip(meta["run_id"], meta["char_idx"]) if int(c) >= 0))
    output = {
        "status": "pass" if all(row["mismatch_count"] == 0 and row["characters_left"] == row["characters_right"] for row in parity_by_alpha.values()) else "fail",
        "population": {"subjects": len(set(r.split("_SE", 1)[0] for r, _ in oof_char_keys)),
                       "runs": len(set(r for r, _ in oof_char_keys)),
                       "run_character_keys_in_cache": len(oof_char_keys),
                       "scored_characters": parity_by_alpha["0.0"]["characters_compared"]},
        "calibration_scope": "clean Study D nested run-grouped OOF; no manifest labels or EEG read",
        "stopping": {"tau": 0.80, "min_sequences": 2, "max_sequences": 10},
        "fusion_alpha_sensitivity": parity_by_alpha,
        "alpha_notes": {
            "0.1": "classifier finalist fixed fusion weight used in prior inner fusion evaluation",
            "0.85": "Study D ablation rung-1 weight fitted on training-pool OOF",
            "0.0": "EEG-only reference",
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, default=Path("results/tables/classifier_nested_calibration_scores.npz"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache/classifier_epochs.h5"))
    parser.add_argument("--output", type=Path, default=Path("results/tables/phase0_oof_evaluator_parity.json"))
    args = parser.parse_args()
    result = run(args.scores, args.cache, args.output)
    print(json.dumps({"status": result["status"], "population": result["population"],
                      "mismatch_by_alpha": {k: v["mismatch_count"] for k, v in result["fusion_alpha_sensitivity"].items()}}, indent=2))


if __name__ == "__main__":
    main()
