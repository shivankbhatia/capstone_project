#!/usr/bin/env python3
"""Grouped OOF evaluation of xDAWN tangent-space LR and Riemannian MDM."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from pyriemann.estimation import XdawnCovariances
from pyriemann.tangentspace import TangentSpace
from pyriemann.classification import MDM
from pyriemann.spatialfilters import Xdawn
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pyriemann

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.screen_shrinkage_lda import (  # noqa: E402
    _character_metrics, _load_metadata, _balanced_fit_indices,
)
from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint, load_heldout_run_ids,
)


def _load_tensor(dataset, indices, cfg, channels, sfreq, tmin, batch_size=1024):
    xs, kept = [], []
    if cfg["channels"] == "subset":
        pick = [channels.index(c) for c in ("Fz", "Cz", "Pz", "PO7", "PO8", "Oz")]
    else:
        pick = list(range(len(channels)))
    times = tmin + np.arange(dataset.shape[-1]) / sfreq
    time_keep = (times >= 0) & (times <= cfg["window_s"] + 1e-8)
    for start in range(0, len(indices), batch_size):
        batch_idx = indices[start:start + batch_size]
        X = dataset[batch_idx][:, pick, :][:, :, time_keep][:, :, ::int(cfg["decim"])]
        keep = np.ones(len(X), dtype=bool)
        if cfg.get("artifact_rejection", False):
            keep = np.ptp(X, axis=-1).max(axis=-1) <= float(cfg["artifact_threshold_uv"]) * 1e-6
        if keep.any():
            xs.append(X[keep].astype(np.float64))
            kept.append(batch_idx[keep])
    return np.concatenate(xs, axis=0), np.concatenate(kept)


def _metrics(oof, meta):
    valid = np.isfinite(oof)
    y, scores = meta["y"][valid], oof[valid]
    per_subject = {}
    for subject in sorted(set(meta["subject"][valid])):
        mask = meta["subject"][valid] == subject
        per_subject[subject] = {
            "epochs": int(mask.sum()),
            "epoch_auc": float(roc_auc_score(y[mask], scores[mask])),
        }
    chars = _character_metrics(
        scores, y, meta["run_id"][valid], meta["char_idx"][valid],
        meta["seq_idx"][valid], meta["stim_code"][valid],
    )
    return {
        "oof_epochs": int(valid.sum()),
        "epoch_auc": float(roc_auc_score(y, scores)),
        "per_subject_epoch_auc": per_subject,
        "character_metrics": chars,
    }


def run(cache_path, config_path, manifest_path, folds, fit_samples, selected_models=None,
        scores_output=None):
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    heldout = load_heldout_run_ids(manifest_path)
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        assert_training_pool_disjoint(
            [f"{run_id}-epo.fif" for run_id in sorted(set(meta["run_id"]))], heldout
        )
        n_folds = min(folds, len(set(meta["run_id"])))
        splitter = GroupKFold(n_splits=n_folds)
        results = {}
        score_arrays = {}
        model_names = selected_models or (
            "M3a_xDAWN_LDA", "M3b_xDAWN_TS_LR", "M3c_Riemannian_MDM"
        )
        for model_name in model_names:
            oof = np.full(len(meta["y"]), np.nan)
            for fold, (train_idx, val_idx) in enumerate(
                splitter.split(np.arange(len(oof)), meta["y"], groups=meta["run_id"]), start=1
            ):
                train_idx = _balanced_fit_indices(
                    train_idx, meta["y"], fit_samples, seed=42 + fold
                )
                X_train, train_idx = _load_tensor(
                    h5["X"], train_idx, cfg, channels, sfreq, tmin
                )
                X_val, val_idx = _load_tensor(
                    h5["X"], val_idx, cfg, channels, sfreq, tmin
                )
                y_train = meta["y"][train_idx]
                if model_name == "M3a_xDAWN_LDA":
                    xdawn = Xdawn(nfilter=4, estimator="scm").fit(X_train, y_train)
                    Z_train = xdawn.transform(X_train).reshape(len(X_train), -1)
                    Z_val = xdawn.transform(X_val).reshape(len(X_val), -1)
                    model = make_pipeline(
                        StandardScaler(),
                        LinearDiscriminantAnalysis(
                            solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]
                        ),
                    ).fit(Z_train, y_train)
                    scores = model.decision_function(Z_val)
                else:
                    covariances = XdawnCovariances(nfilter=4, estimator="scm")
                    C_train = covariances.fit_transform(X_train, y_train)
                    C_val = covariances.transform(X_val)
                if model_name == "M3b_xDAWN_TS_LR":
                    tangent = TangentSpace(metric="riemann")
                    Z_train = tangent.fit_transform(C_train)
                    Z_val = tangent.transform(C_val)
                    model = LogisticRegression(
                        class_weight="balanced", C=1.0, max_iter=500,
                        random_state=42 + fold,
                    ).fit(Z_train, y_train)
                    scores = model.decision_function(Z_val)
                elif model_name == "M3c_Riemannian_MDM":
                    model = MDM(metric="riemann").fit(C_train, y_train)
                    distances = model.transform(C_val)
                    # Positive values indicate the target class is closer.
                    scores = distances[:, 0] - distances[:, 1]
                oof[val_idx] = scores
                print(f"{model_name}: fold {fold}/{n_folds} complete", flush=True)
            results[model_name] = _metrics(oof, meta)
            score_arrays[model_name] = oof
            
        if scores_output is not None:
            scores_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(scores_output, logits_by_model=np.stack(
                [score_arrays[name] for name in score_arrays]),
                model_names=np.asarray(list(score_arrays)), **meta)
    return {
        "cache": str(cache_path),
        "preprocessing": cfg,
        "cv": {"method": "GroupKFold", "groups": "run_id", "folds": n_folds},
        "fit_sample_cap": fit_samples,
        "class_balance": "balanced target/non-target training sample",
        "xdawn_filters": 4,
        "pyriemann_version": pyriemann.__version__,
        "models": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--fit-samples", type=int, default=3000)
    parser.add_argument("--model", action="append", choices=(
        "M3a_xDAWN_LDA", "M3b_xDAWN_TS_LR", "M3c_Riemannian_MDM"
    ), help="Restrict to selected classical candidate; repeat as needed.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/classifier_riemann_inner.json")
    parser.add_argument("--scores-output", type=Path,
                        default=ROOT / "results/tables/classifier_riemann_inner_scores.npz")
    args = parser.parse_args()
    result = run(args.cache, args.config, args.manifest, args.folds,
                 args.fit_samples, args.model, args.scores_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(args.output.read_text()) if args.output.exists() else {}
    prior = all_results.get(args.config.stem, {})
    prior.update({key: value for key, value in result.items() if key != "models"})
    prior.setdefault("models", {}).update(result["models"])
    all_results[args.config.stem] = prior
    args.output.write_text(json.dumps(all_results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"epoch_auc": value["epoch_auc"],
                             "char_acc": value["character_metrics"]["full_repetition_accuracy"]}
                      for name, value in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
