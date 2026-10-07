#!/usr/bin/env python3
"""Nested run-grouped calibration on row/column sequence likelihoods.

For each outer run fold, temperature is fitted only from inner OOF logits
generated within that outer fold's training runs. Evaluation logits come from
an outer model never fitted on those runs. Calibration uses the row and column
softmax likelihood consumed by the decoder, not epoch-level BCE/ECE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pyriemann.estimation import XdawnCovariances  # noqa: E402
from pyriemann.spatialfilters import Xdawn  # noqa: E402
from pyriemann.tangentspace import TangentSpace  # noqa: E402
from scripts.run_classical_cv import _fit_m0_streaming  # noqa: E402
from scripts.run_riemann_cv import _load_tensor  # noqa: E402
from scripts.screen_shrinkage_lda import (  # noqa: E402
    _balanced_fit_indices, _load_metadata, _load_transformed,
)
from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint, load_heldout_run_ids,
)


def _sequence_arrays(logits, y, run_id, char_idx, seq_idx, stim_code):
    records = []
    groups = {}
    for i, key in enumerate(zip(run_id, char_idx, seq_idx)):
        if int(key[1]) >= 0 and int(key[2]) >= 0:
            groups.setdefault((str(key[0]), int(key[1]), int(key[2])), []).append(i)
    for key, rows in groups.items():
        ix = np.asarray(rows, dtype=int)
        codes = stim_code[ix]
        target_codes = codes[y[ix] == 1]
        true_row = target_codes[(target_codes >= 1) & (target_codes <= 9)]
        true_col = target_codes[(target_codes >= 10) & (target_codes <= 17)]
        if len(ix) != 17 or not len(true_row) or not len(true_col):
            continue
        row_scores = np.zeros(9)
        col_scores = np.zeros(8)
        for score, code in zip(logits[ix], codes):
            if 1 <= code <= 9:
                row_scores[code - 1] += score
            elif 10 <= code <= 17:
                col_scores[code - 10] += score
        records.append({
            "key": key, "row_scores": row_scores, "col_scores": col_scores,
            "true_row": int(np.bincount(true_row - 1).argmax()),
            "true_col": int(np.bincount(true_col - 10).argmax()),
        })
    return records


def _sequence_nll(log_temperature, logits, labels, metadata):
    temperature = np.exp(log_temperature)
    records = _sequence_arrays(logits / temperature, labels, *metadata)
    if not records:
        return float("inf")
    losses = []
    for rec in records:
        r, c = rec["row_scores"], rec["col_scores"]
        losses.append(logsumexp(r) - r[rec["true_row"]])
        losses.append(logsumexp(c) - c[rec["true_col"]])
    return float(np.mean(losses))


def _sequence_metrics(logits, labels, metadata):
    records = _sequence_arrays(logits, labels, *metadata)
    row_loss, col_loss, conf, correct = [], [], [], []
    for rec in records:
        for scores, target in ((rec["row_scores"], rec["true_row"]),
                               (rec["col_scores"], rec["true_col"])):
            p = softmax(scores)
            row_loss.append(-np.log(max(float(p[target]), 1e-15)))
            conf.append(float(p.max()))
            correct.append(int(p.argmax() == target))
    conf = np.asarray(conf)
    correct = np.asarray(correct)
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (conf >= low) & ((conf < high) if high < 1 else (conf <= high))
        if mask.any():
            ece += float(mask.mean()) * abs(float(conf[mask].mean() - correct[mask].mean()))
    return {
        "complete_sequences": len(records),
        "row_column_nll": float(np.mean(row_loss)) if row_loss else None,
        "top_label_ece": float(ece),
        "row_accuracy": float(np.mean([r["row_scores"].argmax() == r["true_row"] for r in records])) if records else None,
        "column_accuracy": float(np.mean([r["col_scores"].argmax() == r["true_col"] for r in records])) if records else None,
    }


def _predict(model_name, cache, fit_indices, predict_indices, y, cfg, channels, sfreq, tmin,
             max_fit_samples, seed):
    if model_name == "M0":
        _, _, scores, pred_idx, _ = _fit_m0_streaming(
            cache, fit_indices, predict_indices, y, cfg, channels, sfreq, tmin, seed
        )
        return pred_idx, scores
    fit_idx = _balanced_fit_indices(fit_indices, y, max_fit_samples, seed=seed)
    if model_name == "M2":
        X_train, fit_idx = _load_transformed(cache, fit_idx, cfg, channels, sfreq, tmin)
        X_pred, pred_idx = _load_transformed(cache, predict_indices, cfg, channels, sfreq, tmin)
        model = make_pipeline(
            StandardScaler(), LinearDiscriminantAnalysis(
                solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]
            )
        ).fit(X_train, y[fit_idx])
        return pred_idx, model.decision_function(X_pred)

    X_train, fit_idx = _load_tensor(cache, fit_idx, cfg, channels, sfreq, tmin)
    X_pred, pred_idx = _load_tensor(cache, predict_indices, cfg, channels, sfreq, tmin)
    y_train = y[fit_idx]
    if model_name == "M3b":
        covariances = XdawnCovariances(nfilter=4, estimator="scm")
        C_train = covariances.fit_transform(X_train, y_train)
        C_pred = covariances.transform(X_pred)
        tangent = TangentSpace(metric="riemann")
        Z_train = tangent.fit_transform(C_train)
        Z_pred = tangent.transform(C_pred)
        model = LogisticRegression(
            class_weight="balanced", C=1.0, max_iter=500, random_state=seed,
        ).fit(Z_train, y_train)
        return pred_idx, model.decision_function(Z_pred)
    xdawn = Xdawn(nfilter=4, estimator="scm").fit(X_train, y_train)
    Z_train = xdawn.transform(X_train).reshape(len(X_train), -1)
    Z_pred = xdawn.transform(X_pred).reshape(len(X_pred), -1)
    model = make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(
            solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]
        )
    ).fit(Z_train, y_train)
    return pred_idx, model.decision_function(Z_pred)


def run(cache_path, config_path, manifest_path, model_names, outer_folds=3,
        inner_folds=3, max_fit_samples=3000, output_path=None, scores_path=None):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    heldout = load_heldout_run_ids(manifest_path)
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        run_ids = meta["run_id"]
        assert_training_pool_disjoint([f"{r}-epo.fif" for r in set(run_ids)], heldout)
        outer = GroupKFold(n_splits=min(outer_folds, len(set(run_ids))))
        result = {}
        all_scores = {}
        for model_name in model_names:
            raw_oof = np.full(len(meta["y"]), np.nan)
            calibrated_oof = np.full(len(meta["y"]), np.nan)
            temperatures = []
            for outer_fold, (train_idx, test_idx) in enumerate(
                outer.split(np.arange(len(run_ids)), meta["y"], groups=run_ids), start=1
            ):
                calib_oof = np.full(len(meta["y"]), np.nan)
                inner = GroupKFold(n_splits=min(inner_folds, len(set(run_ids[train_idx]))))
                for inner_fold, (fit_pos, cal_pos) in enumerate(
                    inner.split(train_idx, meta["y"][train_idx], groups=run_ids[train_idx]), start=1
                ):
                    fit_idx, cal_idx = train_idx[fit_pos], train_idx[cal_pos]
                    pred_idx, scores = _predict(
                        model_name, h5["X"], fit_idx, cal_idx, meta["y"], cfg, channels,
                        sfreq, tmin, max_fit_samples, 2000 + outer_fold * 10 + inner_fold,
                    )
                    calib_oof[pred_idx] = scores
                calib_valid = np.isfinite(calib_oof)
                calibration_meta = (
                    run_ids[calib_valid], meta["char_idx"][calib_valid],
                    meta["seq_idx"][calib_valid], meta["stim_code"][calib_valid],
                )
                opt = minimize_scalar(
                    _sequence_nll,
                    bounds=(-4.0, 4.0), method="bounded",
                    args=(calib_oof[calib_valid], meta["y"][calib_valid], calibration_meta),
                )
                temperature = float(np.exp(opt.x))
                temperatures.append(temperature)
                val_idx, val_scores = _predict(
                    model_name, h5["X"], train_idx, test_idx, meta["y"], cfg,
                    channels, sfreq, tmin, max_fit_samples, 3000 + outer_fold,
                )
                raw_oof[val_idx] = val_scores
                calibrated_oof[val_idx] = val_scores / temperature
                print(
                    f"{model_name}: outer fold {outer_fold}/{outer.n_splits}; "
                    f"nested T={temperature:.3f}; {len(val_idx)} validation epochs",
                    flush=True,
                )

            valid = np.isfinite(raw_oof) & (meta["char_idx"] >= 0) & (meta["seq_idx"] >= 0)
            eval_meta = (
                run_ids[valid], meta["char_idx"][valid], meta["seq_idx"][valid],
                meta["stim_code"][valid],
            )
            result[model_name] = {
                "fold_temperatures": temperatures,
                "raw_sequence_metrics": _sequence_metrics(raw_oof[valid], meta["y"][valid], eval_meta),
                "calibrated_sequence_metrics": _sequence_metrics(
                    calibrated_oof[valid], meta["y"][valid], eval_meta
                ),
                "raw_epoch_auc": float(roc_auc_score(meta["y"][np.isfinite(raw_oof)], raw_oof[np.isfinite(raw_oof)])),
                "per_subject_raw_epoch_auc": {
                    subject: float(roc_auc_score(
                        meta["y"][np.isfinite(raw_oof) & (meta["subject"] == subject)],
                        raw_oof[np.isfinite(raw_oof) & (meta["subject"] == subject)],
                    ))
                    for subject in sorted(set(meta["subject"]))
                    if len(np.unique(meta["y"][np.isfinite(raw_oof) & (meta["subject"] == subject)])) == 2
                },
                "per_subject_calibrated_epoch_auc": {
                    subject: float(roc_auc_score(
                        meta["y"][np.isfinite(calibrated_oof) & (meta["subject"] == subject)],
                        calibrated_oof[np.isfinite(calibrated_oof) & (meta["subject"] == subject)],
                    ))
                    for subject in sorted(set(meta["subject"]))
                    if len(np.unique(meta["y"][np.isfinite(calibrated_oof) & (meta["subject"] == subject)])) == 2
                },
                "calibrated_oof_epochs": int(np.isfinite(calibrated_oof).sum()),
                "calibration_protocol": "inner run-grouped OOF within each outer training fold",
            }
            all_scores[model_name + "_raw"] = raw_oof
            all_scores[model_name + "_calibrated"] = calibrated_oof

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if scores_path:
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(scores_path, **all_scores)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    parser.add_argument("--models", nargs="+", choices=("M0", "M2", "M3a", "M3b"), default=("M0", "M2", "M3a"))
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--max-fit-samples", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/classifier_nested_calibration.json")
    parser.add_argument("--scores", type=Path, default=ROOT / "results/tables/classifier_nested_calibration_scores.npz")
    args = parser.parse_args()
    run(args.cache, args.config, args.manifest, args.models, args.outer_folds,
        args.inner_folds, args.max_fit_samples, args.output, args.scores)


if __name__ == "__main__":
    main()
