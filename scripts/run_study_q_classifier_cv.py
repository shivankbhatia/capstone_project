#!/usr/bin/env python3
"""Run the locked M0 recipe on clean Study Q OOF sessions.

The same entry point supports session-grouped CV (primary) and leave-subject-
out CV (secondary). It reads only the clean Q Train cache; it never opens a
ground-truth registry or evaluation vault.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import h5py
import numpy as np
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_classical_cv import _fit_m0_streaming  # noqa: E402
from src.evaluation.classifier_protocol import assert_training_pool_disjoint  # noqa: E402
from src.evaluation.sequence_scoring import aggregate_membership_logits  # noqa: E402

DEFAULT_CACHE = ROOT / "data/cache/study_q_classifier_epochs.h5"
DEFAULT_MANIFEST = ROOT / "splits/study_q_manifest.json"
DEFAULT_EXCLUSIONS = ROOT / "splits/study_q_excluded_training_runs.json"

M0_PREPROCESSING = {
    "channels": "all", "window_s": 0.8, "decim": 1,
    "artifact_rejection": False, "artifact_threshold_uv": 150,
}


def _decode(values) -> np.ndarray:
    return np.asarray([v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in values])


def _character_metrics(logits, meta) -> dict:
    by_character: dict[tuple[str, int], list[int]] = {}
    for index, key in enumerate(zip(meta["run_id"], meta["char_idx"])):
        if key[1] >= 0:
            by_character.setdefault((str(key[0]), int(key[1])), []).append(index)
    per_subject: dict[str, dict[str, int]] = {}
    nll = []
    depth_hist = {}
    invalid_truth = 0
    for (run_id, _), rows in by_character.items():
        indices = np.asarray(rows, dtype=int)
        target_masks = meta["lit_mask"][indices[meta["y"][indices] == 1]]
        if not len(target_masks):
            invalid_truth += 1
            continue
        intersection = np.all(target_masks.astype(bool), axis=0)
        if int(intersection.sum()) != 1:
            invalid_truth += 1
            continue
        true_cell = int(np.flatnonzero(intersection)[0])
        sequences = np.sort(np.unique(meta["seq_idx"][indices]))
        sequences = sequences[sequences >= 0]
        if not len(sequences):
            invalid_truth += 1
            continue
        cumulative = np.zeros(72, dtype=float)
        for sequence in sequences:
            local = indices[meta["seq_idx"][indices] == sequence]
            posterior = aggregate_membership_logits(
                logits[local], meta["lit_mask"][local], n_rows=9, n_cols=8,
            )
            cumulative += np.log(np.clip(posterior, 1e-12, 1.0))
        posterior = softmax(cumulative)
        guess = int(posterior.argmax())
        depth_hist[str(len(sequences))] = depth_hist.get(str(len(sequences)), 0) + 1
        stats = per_subject.setdefault(run_id.split("_SE", 1)[0], {"n": 0, "correct": 0})
        stats["n"] += 1
        stats["correct"] += int(guess == true_cell)
        nll.append(float(-np.log(np.clip(posterior[true_cell], 1e-12, 1.0))))
    n_chars = sum(v["n"] for v in per_subject.values())
    n_correct = sum(v["correct"] for v in per_subject.values())
    return {
        "characters": n_chars,
        "invalid_target_intersections": invalid_truth,
        "full_available_sequence_accuracy": n_correct / n_chars if n_chars else None,
        "full_sequence_grid_nll_uncalibrated": float(np.mean(nll)) if nll else None,
        "available_sequence_depth_histogram": dict(sorted(depth_hist.items(), key=lambda x: int(x[0]))),
        "per_subject": {
            subject: {"characters": value["n"], "accuracy": value["correct"] / value["n"]}
            for subject, value in sorted(per_subject.items())
        },
    }


def run(cache_path: Path, manifest_path: Path, exclusions_path: Path,
        scheme: str = "session", folds: int = 3, scores_path: Path | None = None) -> dict:
    with h5py.File(cache_path, "r") as h5:
        meta = {
            "y": h5["y"][:].astype(np.uint8),
            "run_id": _decode(h5["run_id"][:]),
            "subject": _decode(h5["subject"][:]),
            "session": _decode(h5["session"][:]),
            "char_idx": h5["char_idx"][:].astype(np.int32),
            "seq_idx": h5["seq_idx"][:].astype(np.int32),
            "lit_mask": h5["lit_mask"][:].astype(np.uint8),
        }
        heldout = json.loads(manifest_path.read_text(encoding="utf-8"))
        heldout_ids = {run_id for values in heldout.values() for run_id in values}
        excluded = json.loads(exclusions_path.read_text(encoding="utf-8"))
        excluded_ids = {run_id for values in excluded.values() for run_id in values}
        unique_runs = sorted(set(meta["run_id"]))
        assert_training_pool_disjoint([f"{r}-epo.fif" for r in unique_runs], heldout_ids)
        if set(unique_runs) & excluded_ids or any("_SE003_" in run for run in unique_runs):
            raise ValueError("Study Q CV cache overlaps the held-out session")
        if any("_Test" in run for run in unique_runs):
            raise ValueError("Study Q CV cache includes a Test-labeled run")
        if scheme == "session":
            groups = np.char.add(np.char.add(meta["subject"], "_"), meta["session"])
            splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
        elif scheme == "loso":
            groups = meta["subject"]
            splitter = LeaveOneGroupOut()
        else:
            raise ValueError("scheme must be 'session' or 'loso'")

        channels = str(h5.attrs["channels"]).split(",")
        sfreq, tmin = float(h5.attrs["sfreq"]), float(h5.attrs["tmin"])
        oof = np.full(len(meta["y"]), np.nan, dtype=np.float64)
        fold_records = []
        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(np.arange(len(oof)), meta["y"], groups=groups), start=1
        ):
            _, _, validation_logits, validation_idx, train_epochs = _fit_m0_streaming(
                h5["X"], train_idx, val_idx, meta["y"], M0_PREPROCESSING,
                channels, sfreq, tmin, seed=42 + fold,
            )
            oof[validation_idx] = validation_logits
            fold_records.append({
                "fold": fold,
                "training_subjects": int(len(set(meta["subject"][train_idx]))),
                "validation_subjects": int(len(set(meta["subject"][validation_idx]))),
                "training_sessions": sorted(set(groups[train_idx])),
                "validation_sessions": sorted(set(groups[validation_idx])) if scheme == "session" else [],
                "validation_heldout_subjects": sorted(set(meta["subject"][validation_idx])) if scheme == "loso" else [],
                "training_epochs_after_preprocessing": int(train_epochs),
            })
            print(f"Q M0 {scheme} fold {fold} complete", flush=True)

        valid = np.isfinite(oof)
        y, scores = meta["y"][valid], oof[valid]
        if len(np.unique(y)) != 2:
            raise ValueError("OOF predictions do not contain both classes")
        subject_metrics = {}
        for subject in sorted(set(meta["subject"][valid])):
            mask = valid & (meta["subject"] == subject)
            if len(np.unique(meta["y"][mask])) == 2:
                subject_metrics[subject] = {
                    "epoch_auc": float(roc_auc_score(meta["y"][mask], oof[mask])),
                    "balanced_accuracy_at_zero": float(balanced_accuracy_score(meta["y"][mask], oof[mask] > 0)),
                }
        result = {
            "model": "M0_clean_SGD",
            "study": "StudyQ",
            "scheme": scheme,
            "grouping": "subject-session" if scheme == "session" else "subject",
            "folds": len(fold_records),
            "training_runs": len(unique_runs),
            "heldout_session": "SE003",
            "heldout_test_data_read": False,
            "git_hash": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "excluded_train_sha256": hashlib.sha256(exclusions_path.read_bytes()).hexdigest(),
            "preprocessing": {
                "bandpass_hz": [1.0, 12.0], "window_s": [0.0, 0.8],
                "decimation": 2, "channels": "all", "baseline_correction": False,
                "artifact_rejection": False,
            },
            "recipe": {"classifier": "SGDClassifier(loss=log_loss, penalty=l2, alpha=.01, average=True)",
                       "class_weight": {"0": 1.0, "1": 5.0}, "passes": 5},
            "epoch_auc": float(roc_auc_score(y, scores)),
            "balanced_accuracy_at_zero": float(balanced_accuracy_score(y, scores > 0)),
            "per_subject_epoch_metrics": subject_metrics,
            "character_metrics": _character_metrics(oof[valid], {
                **{k: v[valid] for k, v in meta.items()},
            }),
            "folds_detail": fold_records,
        }
        if scores_path is not None:
            scores_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(scores_path, logits=oof, **meta)
        return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--exclusions", type=Path, default=DEFAULT_EXCLUSIONS)
    parser.add_argument("--scheme", choices=("session", "loso"), default="session")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    result = run(args.cache, args.manifest, args.exclusions, args.scheme, args.folds, args.scores)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in {"per_subject_epoch_metrics", "folds_detail"}}, indent=2))
