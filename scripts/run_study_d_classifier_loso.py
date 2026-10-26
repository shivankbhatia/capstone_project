#!/usr/bin/env python3
"""Nested-calibrated LOSO evaluation of the locked clean Study D M0 classifier.

Each outer fold holds out one subject. Temperature is fit using run-grouped
inner OOF scores from the other subjects only. The frozen manifest is never
loaded as evaluation data by this script; the cache must already be the clean
training-only Study D cache.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.nested_sequence_calibration import _sequence_metrics, _sequence_nll  # noqa: E402
from scripts.run_classical_cv import _fit_m0_streaming  # noqa: E402
from scripts.screen_shrinkage_lda import _load_metadata  # noqa: E402
from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint, load_heldout_run_ids,
)
from src.evaluation.sequence_scoring import SequentialDecoder, aggregate_flash_logits  # noqa: E402


def _decode(values):
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _character_metrics(logits, y, meta, tau=0.80, min_sequences=2, max_sequences=10):
    sequences = {}
    for i, (run, char_idx, seq_idx) in enumerate(
        zip(meta["run_id"], meta["char_idx"], meta["seq_idx"])
    ):
        if char_idx >= 0 and seq_idx >= 0:
            sequences.setdefault((str(run), int(char_idx), int(seq_idx)), []).append(i)

    by_character = {}
    dropped_incomplete = 0
    for key, rows in sequences.items():
        ix = np.asarray(rows, dtype=int)
        evidence = aggregate_flash_logits(logits[ix], meta["stim_code"][ix])
        targets = meta["stim_code"][ix][y[ix] == 1]
        true_rows = targets[(targets >= 1) & (targets <= 9)]
        true_cols = targets[(targets >= 10) & (targets <= 17)]
        if evidence is None or len(ix) != 17 or not len(true_rows) or not len(true_cols):
            dropped_incomplete += 1
            continue
        true_row = int(np.bincount(true_rows - 1, minlength=9).argmax())
        true_col = int(np.bincount(true_cols - 10, minlength=8).argmax())
        by_character.setdefault(key[:2], []).append(
            (key[2], evidence, true_row, true_col)
        )

    records = []
    for key, items in by_character.items():
        items.sort(key=lambda item: item[0])
        row_logp = np.zeros(9)
        col_logp = np.zeros(8)
        decoder = SequentialDecoder(
            72, tau=tau, min_sequences=min_sequences, max_sequences=max_sequences
        )
        true_row, true_col = items[0][2], items[0][3]
        used = 0
        for _, evidence, row_truth, col_truth in items:
            if (row_truth, col_truth) != (true_row, true_col):
                continue
            row_logp += np.log(np.clip(evidence.row, 1e-12, 1.0))
            col_logp += np.log(np.clip(evidence.column, 1e-12, 1.0))
            _, stopped = decoder.add(evidence.grid)
            used += 1
            if stopped or used >= max_sequences:
                break
        # Full-depth decision uses every valid sequence, independently of the
        # sequential decoder's stopping event.
        full_row = np.zeros(9)
        full_col = np.zeros(8)
        for _, evidence, row_truth, col_truth in items:
            if (row_truth, col_truth) == (true_row, true_col):
                full_row += np.log(np.clip(evidence.row, 1e-12, 1.0))
                full_col += np.log(np.clip(evidence.column, 1e-12, 1.0))
        records.append({
            "subject": key[0].split("_SE", 1)[0],
            "full_correct": int(int(full_row.argmax()) == true_row and int(full_col.argmax()) == true_col),
            "stopped_correct": int(decoder.result()["decision_idx"] == true_row * 8 + true_col),
            "sequences_available": len(items),
            "sequences_used": used,
        })

    def summarize(rows):
        if not rows:
            return {"characters": 0, "full_repetition_accuracy": None,
                    "locked_stopping_accuracy": None, "mean_sequences_at_stop": None}
        return {
            "characters": len(rows),
            "full_repetition_accuracy": float(np.mean([r["full_correct"] for r in rows])),
            "locked_stopping_accuracy": float(np.mean([r["stopped_correct"] for r in rows])),
            "mean_sequences_at_stop": float(np.mean([r["sequences_used"] for r in rows])),
            "mean_sequences_available": float(np.mean([r["sequences_available"] for r in rows])),
        }

    per_subject = {}
    for subject in sorted({r["subject"] for r in records}):
        per_subject[subject] = summarize([r for r in records if r["subject"] == subject])
    return {"overall": summarize(records), "per_subject": per_subject,
            "incomplete_sequences_dropped": int(dropped_incomplete)}


def run(cache_path: Path, config_path: Path, manifest_path: Path, inner_folds: int,
        tau: float, min_sequences: int, max_sequences: int, output_path: Path,
        scores_path: Path | None = None):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    heldout_ids = load_heldout_run_ids(manifest_path)
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
        meta["subject"] = _decode(meta["subject"])
        meta["run_id"] = _decode(meta["run_id"])
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        run_ids, subjects = meta["run_id"], meta["subject"]
        assert_training_pool_disjoint([f"{r}-epo.fif" for r in sorted(set(run_ids))], heldout_ids)
        if any("_Test" in r for r in run_ids) or any("Dyn" in r for r in run_ids):
            raise ValueError("LOSO cache must contain only clean Study D RC/Train runs")

        outer = LeaveOneGroupOut()
        raw_oof = np.full(len(meta["y"]), np.nan, dtype=np.float64)
        calibrated_oof = np.full(len(meta["y"]), np.nan, dtype=np.float64)
        fold_records = []
        for fold, (train_idx, test_idx) in enumerate(
            outer.split(np.arange(len(run_ids)), meta["y"], groups=subjects), start=1
        ):
            held_subject = str(subjects[test_idx[0]])
            inner_groups = run_ids[train_idx]
            n_inner = min(inner_folds, len(set(inner_groups)))
            inner = GroupKFold(n_splits=n_inner)
            calibration_logits = np.full(len(meta["y"]), np.nan, dtype=np.float64)
            for inner_fold, (fit_pos, cal_pos) in enumerate(
                inner.split(train_idx, meta["y"][train_idx], groups=inner_groups), start=1
            ):
                fit_idx, cal_idx = train_idx[fit_pos], train_idx[cal_pos]
                _, _, scores, pred_idx, _ = _fit_m0_streaming(
                    h5["X"], fit_idx, cal_idx, meta["y"], cfg, channels, sfreq, tmin,
                    seed=2000 + fold * 100 + inner_fold,
                )
                calibration_logits[pred_idx] = scores
            cal_valid = np.isfinite(calibration_logits)
            cal_meta = (
                run_ids[cal_valid], meta["char_idx"][cal_valid],
                meta["seq_idx"][cal_valid], meta["stim_code"][cal_valid],
            )
            optimum = minimize_scalar(
                _sequence_nll, bounds=(-4.0, 4.0), method="bounded",
                args=(calibration_logits[cal_valid], meta["y"][cal_valid], cal_meta),
            )
            temperature = float(np.exp(optimum.x))
            _, _, scores, pred_idx, training_epochs = _fit_m0_streaming(
                h5["X"], train_idx, test_idx, meta["y"], cfg, channels, sfreq, tmin,
                seed=3000 + fold,
            )
            raw_oof[pred_idx] = scores
            calibrated_oof[pred_idx] = scores / temperature
            fold_records.append({
                "fold": fold, "held_out_subject": held_subject,
                "training_subjects": int(len(set(subjects[train_idx]))),
                "training_runs": int(len(set(run_ids[train_idx]))),
                "training_epochs": int(training_epochs),
                "validation_runs": int(len(set(run_ids[test_idx]))),
                "validation_epochs": int(len(pred_idx)),
                "inner_run_grouped_calibration_folds": int(n_inner),
                "sequence_temperature": temperature,
            })
            print(f"LOSO {fold}/{outer.get_n_splits(groups=subjects)} held={held_subject} T={temperature:.4f}", flush=True)

        valid = np.isfinite(raw_oof)
        seq_valid = valid & (meta["char_idx"] >= 0) & (meta["seq_idx"] >= 0)
        seq_meta = (
            run_ids[seq_valid], meta["char_idx"][seq_valid],
            meta["seq_idx"][seq_valid], meta["stim_code"][seq_valid],
        )
        result = {
            "study": "StudyD",
            "model": "clean M0 streaming SGD",
            "evaluation": "LeaveOneGroupOut grouped by subject; each subject scored only by a model trained on the other subjects",
            "calibration": "per-outer-fold scalar temperature fit to inner run-grouped OOF sequence row/column NLL using outer-training subjects only",
            "stopping_policy": {"tau": tau, "min_sequences": min_sequences, "max_sequences": max_sequences},
            "population": {
                "training_cache_runs": int(len(set(run_ids))),
                "subjects": int(len(set(subjects))),
                "epochs": int(len(meta["y"])),
                "held_out_manifest_sha256": str(h5.attrs["manifest_sha256"]),
                "cache_path": str(cache_path),
                "cache_sha256": _sha256(cache_path),
            },
            "preprocessing": cfg,
            "outer_folds": fold_records,
            "metrics": {
                "oof_epochs": int(valid.sum()),
                "epoch_auc": float(roc_auc_score(meta["y"][valid], raw_oof[valid])),
                "raw_sequence": _sequence_metrics(raw_oof[seq_valid], meta["y"][seq_valid], seq_meta),
                "calibrated_sequence": _sequence_metrics(
                    calibrated_oof[seq_valid], meta["y"][seq_valid], seq_meta
                ),
                "characters": _character_metrics(
                    calibrated_oof[valid], meta["y"][valid],
                    {k: v[valid] for k, v in meta.items() if k in
                     {"run_id", "subject", "char_idx", "seq_idx", "stim_code"}},
                    tau=tau, min_sequences=min_sequences, max_sequences=max_sequences,
                ),
            },
            "provenance": {
                "git_commit": __import__("subprocess").check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
                "git_dirty": bool(__import__("subprocess").check_output(
                    ["git", "status", "--porcelain"], cwd=ROOT, text=True
                ).strip()),
                "cache_only_clean_training_runs": True,
                "manifest_test_runs_loaded": False,
                "manifest_sha256": _sha256(manifest_path),
                "config_sha256": _sha256(config_path),
            },
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if scores_path:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            scores_path, logits_raw=raw_oof, logits_calibrated=calibrated_oof,
            y=meta["y"], subject=meta["subject"], run_id=meta["run_id"],
            char_idx=meta["char_idx"], seq_idx=meta["seq_idx"], stim_code=meta["stim_code"],
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=ROOT / "data/cache/classifier_1_12_no_baseline.h5")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/classifier_optimization/preprocessing/baseline_off.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--tau", type=float, default=0.80)
    parser.add_argument("--min-sequences", type=int, default=2)
    parser.add_argument("--max-sequences", type=int, default=10)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/study_d_classifier_loso.json")
    parser.add_argument("--scores", type=Path, default=ROOT / "results/tables/study_d_classifier_loso_scores.npz")
    args = parser.parse_args()
    result = run(args.cache, args.config, args.manifest, args.inner_folds,
                 args.tau, args.min_sequences, args.max_sequences, args.output, args.scores)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
