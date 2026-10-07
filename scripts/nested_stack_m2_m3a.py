#!/usr/bin/env python3
"""Leakage-safe nested OOF stack of M2 shrinkage LDA and M3a xDAWN-LDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.nested_sequence_calibration import (  # noqa: E402
    _load_metadata, _predict, _sequence_metrics, _sequence_nll,
)
from scripts.screen_shrinkage_lda import _decode_strings  # noqa: E402
from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint, load_heldout_run_ids,
)


def _fit_base_oof(cache, pool, y, run_ids, cfg, channels, sfreq, tmin,
                  max_fit_samples, folds, seed):
    outputs = [np.full(len(y), np.nan), np.full(len(y), np.nan)]
    splitter = GroupKFold(n_splits=min(folds, len(set(run_ids[pool]))))
    for fold, (fit_pos, pred_pos) in enumerate(
        splitter.split(pool, y[pool], groups=run_ids[pool]), start=1
    ):
        fit_idx, pred_idx = pool[fit_pos], pool[pred_pos]
        indices = []
        for model_pos, model_name in enumerate(("M2", "M3a")):
            idx, logits = _predict(
                model_name, cache, fit_idx, pred_idx, y, cfg, channels,
                sfreq, tmin, max_fit_samples, seed + fold * 100 + model_pos,
            )
            indices.append(idx)
            outputs[model_pos][idx] = logits
        if not np.array_equal(indices[0], indices[1]):
            raise RuntimeError("M2/M3a OOF prediction indices differ")
    return outputs


def _aligned_features(indices, a, b):
    if not np.array_equal(indices, indices):  # explicit alignment invariant
        raise RuntimeError("base score indices do not align")
    valid = np.isfinite(a[indices]) & np.isfinite(b[indices])
    return indices[valid], np.column_stack((a[indices][valid], b[indices][valid]))


def _fit_stacker(indices, score_a, score_b, y, C):
    idx, X = _aligned_features(indices, score_a, score_b)
    if len(idx) == 0 or len(np.unique(y[idx])) != 2:
        raise RuntimeError("stacker training fold lacks both classes")
    clf = LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                             class_weight="balanced", max_iter=1000)
    clf.fit(X, y[idx])
    return clf


def run(cache_path, config_path, manifest_path, output_path, scores_path,
        outer_folds=3, inner_folds=3, max_fit_samples=3000, C=0.1):
    cfg = json.loads(config_path.read_text())
    heldout = load_heldout_run_ids(manifest_path)
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
        meta["run_id"] = _decode_strings(meta["run_id"])
        meta["subject"] = _decode_strings(meta["subject"])
        y = meta["y"]
        run_ids = meta["run_id"]
        assert_training_pool_disjoint([f"{r}-epo.fif" for r in set(run_ids)], heldout)
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        n = len(y)
        raw_oof = np.full(n, np.nan)
        calibrated_oof = np.full(n, np.nan)
        fold_info = []
        outer = GroupKFold(n_splits=min(outer_folds, len(set(run_ids))))
        for outer_fold, (train_pos, test_pos) in enumerate(
            outer.split(np.arange(n), y, groups=run_ids), start=1
        ):
            train_idx, test_idx = train_pos, test_pos

            # Produce disjoint inner-train stacked scores for temperature fit.
            inner_stacked = np.full(n, np.nan)
            inner = GroupKFold(n_splits=min(inner_folds, len(set(run_ids[train_idx]))))
            for inner_fold, (fit_pos, val_pos) in enumerate(
                inner.split(train_idx, y[train_idx], groups=run_ids[train_idx]), start=1
            ):
                stack_train, stack_val = train_idx[fit_pos], train_idx[val_pos]
                train_a, train_b = _fit_base_oof(
                    h5["X"], stack_train, y, run_ids, cfg, channels, sfreq, tmin,
                    max_fit_samples, inner_folds, 10000 + outer_fold * 1000 + inner_fold * 100,
                )
                stacker = _fit_stacker(stack_train, train_a, train_b, y, C)
                val_logits = []
                val_ix = None
                for model_pos, model_name in enumerate(("M2", "M3a")):
                    ix, scores = _predict(
                        model_name, h5["X"], stack_train, stack_val, y, cfg,
                        channels, sfreq, tmin, max_fit_samples,
                        20000 + outer_fold * 1000 + inner_fold * 10 + model_pos,
                    )
                    if val_ix is None:
                        val_ix = ix
                    elif not np.array_equal(val_ix, ix):
                        raise RuntimeError("M2/M3a inner validation indices differ")
                    val_logits.append(scores)
                if val_ix is None:
                    raise RuntimeError("empty inner validation fold")
                inner_stacked[val_ix] = stacker.decision_function(
                    np.column_stack(val_logits)
                )

            cal_idx = train_idx[np.isfinite(inner_stacked[train_idx])]
            cal_meta = (run_ids[cal_idx], meta["char_idx"][cal_idx],
                        meta["seq_idx"][cal_idx], meta["stim_code"][cal_idx])
            temp_fit = minimize_scalar(
                _sequence_nll, bounds=(-4, 4), method="bounded",
                args=(inner_stacked[cal_idx], y[cal_idx], cal_meta),
            )
            temperature = float(np.exp(temp_fit.x))

            # Refit stacker on all outer-training OOF scores, evaluate outer fold.
            outer_a, outer_b = _fit_base_oof(
                h5["X"], train_idx, y, run_ids, cfg, channels, sfreq, tmin,
                max_fit_samples, inner_folds, 30000 + outer_fold * 1000,
            )
            stacker = _fit_stacker(train_idx, outer_a, outer_b, y, C)
            pred_features = []
            pred_idx = None
            for model_pos, model_name in enumerate(("M2", "M3a")):
                ix, scores = _predict(
                    model_name, h5["X"], train_idx, test_idx, y, cfg,
                    channels, sfreq, tmin, max_fit_samples,
                    40000 + outer_fold * 10 + model_pos,
                )
                if pred_idx is None:
                    pred_idx = ix
                elif not np.array_equal(pred_idx, ix):
                    raise RuntimeError("M2/M3a outer prediction indices differ")
                pred_features.append(scores)
            raw = stacker.decision_function(np.column_stack(pred_features))
            raw_oof[pred_idx] = raw
            calibrated_oof[pred_idx] = raw / temperature
            fold_info.append({"fold": outer_fold, "temperature": temperature,
                              "stacker_intercept": float(stacker.intercept_[0]),
                              "stacker_coefficients": stacker.coef_[0].tolist(),
                              "train_runs": len(set(run_ids[train_idx])),
                              "validation_runs": len(set(run_ids[test_idx]))})
            print(f"stack outer fold {outer_fold}/{outer.n_splits}; T={temperature:.3f}", flush=True)

    valid = np.isfinite(raw_oof) & (meta["char_idx"] >= 0) & (meta["seq_idx"] >= 0)
    indices = np.flatnonzero(valid)
    meta_tuple = (run_ids[valid], meta["char_idx"][valid], meta["seq_idx"][valid],
                  meta["stim_code"][valid])
    result = {"M2_M3a_stack": {
        "folds": fold_info,
        "raw_sequence_metrics": _sequence_metrics(raw_oof[valid], y[valid], meta_tuple),
        "calibrated_sequence_metrics": _sequence_metrics(calibrated_oof[valid], y[valid], meta_tuple),
        "calibration_protocol": "nested run-grouped cross-fit of stacker predictions; sequence row/column NLL temperature",
        "stacker": {"type": "L2 logistic regression", "C": C, "class_weight": "balanced"},
    }}
    score_arrays = {"M2_M3a_stack_raw": raw_oof,
                    "M2_M3a_stack_calibrated": calibrated_oof}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(scores_path, **score_arrays)
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    p.add_argument("--output", type=Path, default=ROOT / "results/tables/classifier_stack.json")
    p.add_argument("--scores", type=Path, default=ROOT / "results/tables/classifier_stack_scores.npz")
    p.add_argument("--outer-folds", type=int, default=3)
    p.add_argument("--inner-folds", type=int, default=3)
    p.add_argument("--max-fit-samples", type=int, default=3000)
    p.add_argument("--C", type=float, default=0.1)
    a = p.parse_args()
    run(a.cache, a.config, a.manifest, a.output, a.scores,
        a.outer_folds, a.inner_folds, a.max_fit_samples, a.C)


if __name__ == "__main__":
    main()
