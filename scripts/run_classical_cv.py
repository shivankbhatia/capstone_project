#!/usr/bin/env python3
"""Run clean run-grouped OOF evaluation for M0 and tuned stepwise M1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.screen_shrinkage_lda import (  # noqa: E402
    _character_metrics,
    _decode_strings,
    _load_metadata,
    _load_transformed,
    _balanced_fit_indices,
)
from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint,
    load_heldout_run_ids,
)
from src.models.swlda import StepwiseLinearDiscriminant  # noqa: E402


STEPWISE_GRID = ((0.01, 0.10), (0.05, 0.10), (0.05, 0.20))


def _fit_indices(indices, y, max_samples, seed, balanced):
    if balanced:
        return _balanced_fit_indices(indices, y, max_samples, seed)
    if max_samples <= 0 or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    local_y = y[indices]
    classes, counts = np.unique(local_y, return_counts=True)
    proportions = counts / counts.sum()
    selected = []
    remaining = max_samples
    for i, cls in enumerate(classes):
        take = remaining if i == len(classes) - 1 else int(round(max_samples * proportions[i]))
        members = indices[local_y == cls]
        take = min(take, len(members))
        selected.append(rng.choice(members, size=take, replace=False))
        remaining -= take
    return np.sort(np.concatenate(selected))


def _new_m0(seed: int):
    return SGDClassifier(
        loss="log_loss", penalty="l2", alpha=0.01, average=True,
        class_weight={0: 1.0, 1: 5.0}, random_state=seed,
    )


def _fit_m0_streaming(dataset, train_idx, val_idx, y, cfg, channels, sfreq, tmin, seed):
    """Five streaming passes over full training runs, matching the legacy recipe."""
    scaler = StandardScaler()
    kept_train = []
    for start in range(0, len(train_idx), 2048):
        batch_idx = train_idx[start:start + 2048]
        X = dataset[batch_idx]
        features, keep = _load_transform_batch(X, cfg, channels, sfreq, tmin)
        if keep.any():
            scaler.partial_fit(features[keep])
            kept_train.append(batch_idx[keep])
    kept_train = np.concatenate(kept_train)

    classifier = _new_m0(seed)
    first_batch = True
    for _ in range(5):
        for start in range(0, len(kept_train), 2048):
            batch_idx = kept_train[start:start + 2048]
            features, keep = _load_transform_batch(
                dataset[batch_idx], cfg, channels, sfreq, tmin
            )
            if not keep.any():
                continue
            scaled = scaler.transform(features[keep])
            classifier.partial_fit(
                scaled, y[batch_idx[keep]], classes=[0, 1] if first_batch else None
            )
            first_batch = False

    val_scores = []
    kept_val = []
    for start in range(0, len(val_idx), 2048):
        batch_idx = val_idx[start:start + 2048]
        features, keep = _load_transform_batch(
            dataset[batch_idx], cfg, channels, sfreq, tmin
        )
        if keep.any():
            val_scores.append(classifier.decision_function(scaler.transform(features[keep])))
            kept_val.append(batch_idx[keep])
    return classifier, scaler, np.concatenate(val_scores), np.concatenate(kept_val), len(kept_train)


def _load_transform_batch(X, cfg, channels, sfreq, tmin):
    from scripts.screen_shrinkage_lda import _transform
    return _transform(X, cfg, channels, sfreq, tmin)


def _fit_m1(X, y, params):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = StepwiseLinearDiscriminant(
        p_enter=params[0], p_remove=params[1], max_features=25,
        class_weight={0: 1.0, 1: 7.0},
    ).fit(X_scaled, y)
    return scaler, model


def _predict_m1(fitted, X):
    scaler, model = fitted
    return model.decision_function(scaler.transform(X))


def _tune_stepwise(X, y, groups, seed):
    unique = np.unique(groups)
    n_splits = min(2, len(unique))
    if n_splits < 2:
        return STEPWISE_GRID[1], {"note": "single run group in training fold"}
    splitter = GroupKFold(n_splits=n_splits)
    scores = {}
    for params in STEPWISE_GRID:
        fold_auc = []
        for fold, (fit_idx, val_idx) in enumerate(splitter.split(X, y, groups), start=1):
            if len(np.unique(y[fit_idx])) != 2 or len(np.unique(y[val_idx])) != 2:
                continue
            fitted = _fit_m1(X[fit_idx], y[fit_idx], params)
            fold_auc.append(roc_auc_score(y[val_idx], _predict_m1(fitted, X[val_idx])))
        scores["{:.3f}/{:.3f}".format(*params)] = float(np.mean(fold_auc)) if fold_auc else 0.0
    best_key = max(scores, key=scores.get)
    enter, remove = [float(v) for v in best_key.split("/")]
    return (enter, remove), scores


def _metrics(oof, meta):
    valid = np.isfinite(oof)
    y, scores = meta["y"][valid], oof[valid]
    subjects = meta["subject"][valid]
    per_subject = {}
    for subject in sorted(set(subjects)):
        mask = subjects == subject
        per_subject[subject] = {
            "epoch_auc": float(roc_auc_score(y[mask], scores[mask])),
            "epochs": int(mask.sum()),
        }
    chars = _character_metrics(
        scores, y, meta["run_id"][valid], meta["char_idx"][valid],
        meta["seq_idx"][valid], meta["stim_code"][valid],
    )
    return {
        "epoch_auc": float(roc_auc_score(y, scores)),
        "per_subject_epoch_auc": per_subject,
        "character_metrics": chars,
        "oof_epochs": int(valid.sum()),
    }


def run(cache_path: Path, config_path: Path, manifest_path: Path,
        folds: int, max_fit_samples: int, scores_output: Path | None = None) -> dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    heldout = load_heldout_run_ids(manifest_path)
    score_arrays = {}
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        assert_training_pool_disjoint(
            [f"{run_id}-epo.fif" for run_id in sorted(set(meta["run_id"]))], heldout
        )
        n_folds = min(folds, len(set(meta["run_id"])))
        splitter = GroupKFold(n_splits=n_folds)
        out = {}
        for name in ("M0_clean_SGD", "M1_tuned_SWLDA"):
            oof = np.full(len(meta["y"]), np.nan, dtype=np.float64)
            selected_params = []
            for fold, (train_idx, val_idx) in enumerate(
                splitter.split(np.arange(len(oof)), meta["y"], groups=meta["run_id"]), start=1
            ):
                if name == "M0_clean_SGD":
                    model, scaler, val_scores, val_idx, train_count = _fit_m0_streaming(
                        h5["X"], train_idx, val_idx, meta["y"], cfg,
                        channels, sfreq, tmin, seed=42 + fold,
                    )
                    oof[val_idx] = val_scores
                    params = {"partial_fit_epochs": 5, "alpha": 0.01,
                              "positive_class_weight": 5.0,
                              "training_epochs": train_count}
                else:
                    fit_idx = _fit_indices(
                        train_idx, meta["y"], max_fit_samples, seed=42 + fold,
                        balanced=False,
                    )
                    X_fit, fit_idx = _load_transformed(
                        h5["X"], fit_idx, cfg, channels, sfreq, tmin
                    )
                    X_val, val_idx = _load_transformed(
                        h5["X"], val_idx, cfg, channels, sfreq, tmin
                    )
                    y_fit = meta["y"][fit_idx]
                    params, inner = _tune_stepwise(
                        X_fit, y_fit, meta["run_id"][fit_idx], seed=42 + fold
                    )
                    model = _fit_m1(X_fit, y_fit, params)
                    oof[val_idx] = _predict_m1(model, X_val)
                    params = {"p_enter": params[0], "p_remove": params[1],
                              "max_features": 25, "inner_fold_auc": inner}
                selected_params.append({"outer_fold": fold, "selected": params})
                print(f"{name}: fold {fold}/{n_folds} complete", flush=True)
            out[name] = {"metrics": _metrics(oof, meta), "selected_parameters": selected_params}
            score_arrays[name] = oof

        if scores_output is not None:
            scores_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(scores_output, logits_by_model=np.stack(
                [score_arrays[name] for name in score_arrays]),
                model_names=np.asarray(list(score_arrays)), **meta)

    return {
        "cache": str(cache_path),
        "preprocessing": cfg,
        "cv": {"method": "GroupKFold", "groups": "run_id", "folds": n_folds},
        "max_fit_samples_per_fold": max_fit_samples,
        "git_hash": __import__("subprocess").check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(__import__("subprocess").check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
        "models": out,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--max-fit-samples", type=int, default=3000)
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/classifier_classical_inner.json")
    parser.add_argument("--scores-output", type=Path,
                        default=ROOT / "results/tables/classifier_classical_inner_scores.npz")
    args = parser.parse_args()
    result = run(args.cache, args.config, args.manifest, args.folds, args.max_fit_samples,
                 args.scores_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    output[args.config.stem] = result
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: model["metrics"] for name, model in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
