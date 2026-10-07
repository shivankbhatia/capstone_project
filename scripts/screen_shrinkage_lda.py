#!/usr/bin/env python3
"""Run run-grouped CV for the preprocessing probe (shrinkage LDA)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from scipy.special import softmax
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import sklearn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint,
    load_heldout_run_ids,
)


CHANNEL_SUBSET = ("Fz", "Cz", "Pz", "PO7", "PO8", "Oz")


def _decode_strings(values) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ])


def _candidate_scores(logits: np.ndarray, codes: np.ndarray) -> np.ndarray:
    rows = np.zeros(9, dtype=np.float64)
    cols = np.zeros(8, dtype=np.float64)
    for logit, code in zip(logits, codes):
        if 1 <= code <= 9:
            rows[code - 1] += logit
        elif 10 <= code <= 17:
            cols[code - 10] += logit
    return (rows[:, None] + cols[None, :]).reshape(-1)


def _character_metrics(
    logits: np.ndarray, y: np.ndarray, run_ids: np.ndarray,
    char_idx: np.ndarray, seq_idx: np.ndarray, stim_code: np.ndarray,
) -> dict:
    by_char: dict[tuple[str, int], list[int]] = {}
    for i, key in enumerate(zip(run_ids, char_idx)):
        if key[1] >= 0:
            by_char.setdefault((str(key[0]), int(key[1])), []).append(i)

    full_correct = stopped_correct = 0
    stopped_sequences = []
    per_subject: dict[str, dict[str, int]] = {}
    prefix_correct = {2: 0, 5: 0, 10: 0, 20: 0}
    prefix_total = 0
    for (run_id, _), rows in by_char.items():
        ix = np.asarray(rows, dtype=int)
        positive = stim_code[ix][y[ix] == 1]
        target_rows = positive[(positive >= 1) & (positive <= 9)]
        target_cols = positive[(positive >= 10) & (positive <= 17)]
        if target_rows.size == 0 or target_cols.size == 0:
            continue
        true_row = int(np.bincount(target_rows - 1).argmax())
        true_col = int(np.bincount(target_cols - 10).argmax())
        true_cell = true_row * 8 + true_col

        order = np.unique(seq_idx[ix])
        order = np.sort(order[order >= 0])
        cumulative = np.zeros(72, dtype=np.float64)
        stopped_at = None
        stopped_cell = None
        last_cell = 0
        for step, seq in enumerate(order, start=1):
            srows = ix[seq_idx[ix] == seq]
            cumulative += _candidate_scores(logits[srows], stim_code[srows])
            last_cell = int(np.argmax(cumulative))
            posterior = softmax(cumulative)
            if stopped_at is None and step >= 2 and step <= 10 and posterior.max() >= 0.80:
                stopped_at = step
                stopped_cell = last_cell
            elif stopped_at is None and step == 10:
                stopped_at = step
                stopped_cell = last_cell
        if stopped_at is None:
            stopped_at = min(len(order), 10)
            stopped_cell = last_cell

        full_correct += int(last_cell == true_cell)
        stopped_correct += int(stopped_cell == true_cell)
        stopped_sequences.append(stopped_at)
        subject = run_id.split("_SE", 1)[0]
        stats = per_subject.setdefault(subject, {"characters": 0, "correct": 0})
        stats["characters"] += 1
        stats["correct"] += int(last_cell == true_cell)

        for n_seq in prefix_correct:
            available = order[:n_seq]
            if not len(available):
                continue
            selected = ix[np.isin(seq_idx[ix], available)]
            guess = int(np.argmax(_candidate_scores(logits[selected], stim_code[selected])))
            prefix_correct[n_seq] += int(guess == true_cell)
        prefix_total += 1

    n_chars = len(stopped_sequences)
    return {
        "characters": n_chars,
        "full_repetition_accuracy": full_correct / n_chars if n_chars else None,
        "locked_stopping_accuracy": stopped_correct / n_chars if n_chars else None,
        "mean_sequences_at_stop": float(np.mean(stopped_sequences)) if stopped_sequences else None,
        "prefix_accuracy": {
            str(n): prefix_correct[n] / prefix_total if prefix_total else None
            for n in prefix_correct
        },
        "per_subject": {
            subject: {
                "characters": values["characters"],
                "full_repetition_accuracy": values["correct"] / values["characters"],
            }
            for subject, values in sorted(per_subject.items())
        },
    }


def _load_metadata(h5: h5py.File) -> dict[str, np.ndarray]:
    return {
        "y": h5["y"][:].astype(np.uint8),
        "subject": _decode_strings(h5["subject"][:]),
        "run_id": _decode_strings(h5["run_id"][:]),
        "seq_idx": h5["seq_idx"][:].astype(np.int32),
        "stim_code": h5["stim_code"][:].astype(np.int32),
        "char_idx": h5["char_idx"][:].astype(np.int32),
    }


def _transform(X: np.ndarray, cfg: dict, channels: list[str], sfreq: float, tmin: float):
    if cfg["channels"] == "subset":
        missing = set(CHANNEL_SUBSET).difference(channels)
        if missing:
            raise ValueError(f"Requested channels absent from cache: {sorted(missing)}")
        pick = [channels.index(name) for name in CHANNEL_SUBSET]
        X = X[:, pick, :]

    times = tmin + np.arange(X.shape[-1]) / sfreq
    keep = (times >= 0.0) & (times <= cfg["window_s"] + 1e-8)
    X = X[:, :, keep]
    X = X[:, :, ::int(cfg["decim"])]
    artifact_keep = np.ones(len(X), dtype=bool)
    if cfg["artifact_rejection"]:
        peak_to_peak = np.ptp(X, axis=-1).max(axis=-1)
        artifact_keep = peak_to_peak <= float(cfg["artifact_threshold_uv"]) * 1e-6
    return X.reshape(len(X), -1), artifact_keep


def _load_transformed(dataset, indices, cfg, channels, sfreq, tmin, batch_size=2048):
    feature_chunks = []
    kept_indices = []
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        X = dataset[selected]
        features, keep = _transform(X, cfg, channels, sfreq, tmin)
        feature_chunks.append(features[keep])
        kept_indices.append(selected[keep])
    return np.concatenate(feature_chunks, axis=0), np.concatenate(kept_indices)


def _balanced_fit_indices(indices, y, max_samples, seed):
    if max_samples <= 0 or len(indices) <= max_samples:
        return indices
    rng = np.random.default_rng(seed)
    labels = y[indices]
    classes = np.unique(labels)
    if len(classes) != 2:
        raise ValueError("Each training fold must contain both target classes")
    per_class = max_samples // 2
    selected = []
    for cls in classes:
        members = indices[labels == cls]
        take = min(per_class, len(members))
        selected.append(rng.choice(members, size=take, replace=False))
    return np.sort(np.concatenate(selected))


def run_screen(
    cache_path: Path, cfg: dict, n_splits: int, manifest_path: Path,
    max_fit_samples: int, scores_output: Path | None = None,
) -> dict:
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        prep_source = {
            "bandpass_hz": [float(h5.attrs.get("l_freq", 0.1)), float(h5.attrs.get("h_freq", 30.0))],
            "baseline_correction": bool(h5.attrs.get("baseline_correction", True)),
        }
        unique_runs = sorted(set(meta["run_id"]))
        heldout = load_heldout_run_ids(manifest_path)
        assert_training_pool_disjoint(
            [f"{run_id}-epo.fif" for run_id in unique_runs], heldout
        )
        n_folds = min(n_splits, len(unique_runs))
        splitter = GroupKFold(n_splits=n_folds)
        oof = np.full(len(meta["y"]), np.nan, dtype=np.float64)
        row_indices = np.arange(len(oof))
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(row_indices, meta["y"], groups=meta["run_id"]), start=1
        ):
            train_idx = _balanced_fit_indices(
                train_idx, meta["y"], max_fit_samples, seed=42 + fold
            )
            X_train, train_idx = _load_transformed(
                h5["X"], train_idx, cfg, channels, sfreq, tmin
            )
            X_val, val_idx = _load_transformed(
                h5["X"], val_idx, cfg, channels, sfreq, tmin
            )
            model = make_pipeline(
                StandardScaler(),
                LinearDiscriminantAnalysis(
                    solver="lsqr", shrinkage="auto", priors=[0.5, 0.5]
                ),
            )
            model.fit(X_train, meta["y"][train_idx])
            oof[val_idx] = model.decision_function(X_val)
            print(f"{cfg['name']}: fold {fold}/{n_folds} scored {len(val_idx)} epochs", flush=True)

        if scores_output is not None:
            scores_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(scores_output, logits=oof, **meta)

    valid = np.isfinite(oof)
    y = meta["y"][valid]
    scores = oof[valid]
    subjects = meta["subject"][valid]
    overall = {
        "epochs": int(valid.sum()),
        "targets": int((y == 1).sum()),
        "non_targets": int((y == 0).sum()),
        "epoch_auc": float(roc_auc_score(y, scores)),
        "balanced_accuracy_at_zero": float(balanced_accuracy_score(y, scores >= 0)),
    }
    per_subject = {}
    for subject in sorted(set(subjects)):
        mask = subjects == subject
        sy, ss = y[mask], scores[mask]
        per_subject[subject] = {
            "epochs": int(mask.sum()),
            "epoch_auc": float(roc_auc_score(sy, ss)),
            "balanced_accuracy_at_zero": float(balanced_accuracy_score(sy, ss >= 0)),
        }
    char_metrics = _character_metrics(
        oof[valid], meta["y"][valid], meta["run_id"][valid],
        meta["char_idx"][valid], meta["seq_idx"][valid], meta["stim_code"][valid],
    )
    git_hash = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    git_dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip())
    return {
        "model": "M2 shrinkage LDA probe",
        "cache": str(cache_path.relative_to(ROOT)) if cache_path.is_relative_to(ROOT) else str(cache_path),
        "config": cfg,
        "cv": {"method": "GroupKFold", "group": "run_id", "folds": n_folds},
        "seed": 42,
        "max_fit_samples_per_fold": max_fit_samples,
        "training_sampling": "balanced target/non-target subsample; complete validation folds",
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "sklearn_version": sklearn.__version__,
        "preprocessing_source": prep_source,
        "overall": overall,
        "per_subject": per_subject,
        "character_metrics": char_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True,
                        help="JSON config with name, window_s, decim, channels, artifact_rejection")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-fit-samples", type=int, default=5000,
                        help="Balanced target/non-target sample cap per fit; validation scores all epochs.")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/classifier_phase2_screens.json")
    parser.add_argument("--scores-output", type=Path)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    result = run_screen(args.cache, cfg, args.folds, args.manifest, args.max_fit_samples,
                        args.scores_output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(args.output.read_text(encoding="utf-8")) if args.output.exists() else {}
    existing[cfg["name"]] = result
    args.output.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": cfg, "overall": result["overall"],
                      "character_metrics": result["character_metrics"]}, indent=2))


if __name__ == "__main__":
    main()
