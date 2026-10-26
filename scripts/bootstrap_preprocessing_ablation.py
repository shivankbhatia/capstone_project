#!/usr/bin/env python3
"""Paired run-cluster bootstrap for two preprocessing OOF score files."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.screen_shrinkage_lda import _character_metrics  # noqa: E402


PAIRED_KEYS = ("y", "run_id", "subject", "seq_idx", "stim_code", "char_idx")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_scores(path: Path) -> tuple[dict[str, np.ndarray], np.ndarray, str]:
    with np.load(path, allow_pickle=False) as data:
        names = data["model_names"].astype(str)
        if names.tolist() != ["M1_tuned_SWLDA"]:
            raise ValueError(f"Expected only M1_tuned_SWLDA in {path}; got {names.tolist()}")
        meta = {key: data[key].copy() for key in PAIRED_KEYS}
        scores = data["logits_by_model"][0].copy()
    if len(scores) != len(meta["y"]) or not np.isfinite(scores).all():
        raise ValueError(f"Non-finite or mismatched OOF scores in {path}")
    return meta, scores, _sha256(path)


def _run_char_counts(meta: dict[str, np.ndarray], scores: np.ndarray,
                     run_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    correct_counts, char_counts = [], []
    for run_id in run_ids:
        ix = np.flatnonzero(meta["run_id"] == run_id)
        local = {key: values[ix] for key, values in meta.items()}
        metric = _character_metrics(
            scores[ix], local["y"], local["run_id"], local["char_idx"],
            local["seq_idx"], local["stim_code"],
        )
        n_chars = int(metric["characters"])
        char_counts.append(n_chars)
        correct_counts.append(int(round(metric["locked_stopping_accuracy"] * n_chars)))
    return np.asarray(correct_counts, dtype=np.int64), np.asarray(char_counts, dtype=np.int64)


def _pairwise_auc_numerators(run_scores: list[np.ndarray],
                            run_y: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-run-pair positive-over-negative counts (ties count half)."""
    n_runs = len(run_scores)
    wins = np.zeros((n_runs, n_runs), dtype=np.float64)
    positives = np.zeros(n_runs, dtype=np.int64)
    negatives = np.zeros(n_runs, dtype=np.int64)
    sorted_negatives = []
    for i, (scores, labels) in enumerate(zip(run_scores, run_y)):
        positives[i] = int(np.sum(labels == 1))
        negatives[i] = int(np.sum(labels == 0))
        if not positives[i] or not negatives[i]:
            raise ValueError("Every run must contain both target classes")
        sorted_negatives.append(np.sort(scores[labels == 0]))
    for i, (scores, labels) in enumerate(zip(run_scores, run_y)):
        pos_scores = scores[labels == 1]
        for j, neg_scores in enumerate(sorted_negatives):
            left = np.searchsorted(neg_scores, pos_scores, side="left")
            right = np.searchsorted(neg_scores, pos_scores, side="right")
            wins[i, j] = left.sum() + 0.5 * (right - left).sum()
    return wins, positives, negatives


def run(a_path: Path, b_path: Path, output: Path, n_boot: int, seed: int,
        reference_label: str, candidate_label: str) -> dict:
    meta_a, scores_a, hash_a = _load_scores(a_path)
    meta_b, scores_b, hash_b = _load_scores(b_path)
    for key in PAIRED_KEYS:
        if not np.array_equal(meta_a[key], meta_b[key]):
            raise ValueError(f"Score files are not event-paired: metadata differs for {key}")

    run_ids = sorted(np.unique(meta_a["run_id"]).tolist())
    run_indices = [np.flatnonzero(meta_a["run_id"] == run_id) for run_id in run_ids]
    run_y = [meta_a["y"][ix] for ix in run_indices]
    run_a = [scores_a[ix] for ix in run_indices]
    run_b = [scores_b[ix] for ix in run_indices]
    wins_a, positives, negatives = _pairwise_auc_numerators(run_a, run_y)
    wins_b, positives_b, negatives_b = _pairwise_auc_numerators(run_b, run_y)
    if not np.array_equal(positives, positives_b) or not np.array_equal(negatives, negatives_b):
        raise ValueError("Class counts differ between paired score files")

    correct_a, chars_a = _run_char_counts(meta_a, scores_a, run_ids)
    correct_b, chars_b = _run_char_counts(meta_b, scores_b, run_ids)
    if not np.array_equal(chars_a, chars_b):
        raise ValueError("Character counts differ by run across paired score files")

    class_denominator = int(positives.sum() * negatives.sum())
    point_auc_a = float(wins_a.sum() / class_denominator)
    point_auc_b = float(wins_b.sum() / class_denominator)
    if not (np.isclose(point_auc_a, roc_auc_score(meta_a["y"], scores_a))
            and np.isclose(point_auc_b, roc_auc_score(meta_b["y"], scores_b))):
        raise RuntimeError("Run-pair AUC aggregation does not match pooled sklearn AUC")
    point_acc_a = float(correct_a.sum() / chars_a.sum())
    point_acc_b = float(correct_b.sum() / chars_b.sum())

    rng = np.random.default_rng(seed)
    delta_auc = np.empty(n_boot, dtype=np.float64)
    delta_acc = np.empty(n_boot, dtype=np.float64)
    n_runs = len(run_ids)
    for boot in range(n_boot):
        sampled = rng.integers(0, n_runs, size=n_runs)
        multiplicity = np.bincount(sampled, minlength=n_runs)
        denominator = int((multiplicity @ positives) * (multiplicity @ negatives))
        delta_auc[boot] = (
            multiplicity @ wins_b @ multiplicity
            - multiplicity @ wins_a @ multiplicity
        ) / denominator
        total_chars = int(multiplicity @ chars_a)
        delta_acc[boot] = (
            multiplicity @ correct_b - multiplicity @ correct_a
        ) / total_chars

    report = {
        "comparison": f"candidate {candidate_label} minus reference {reference_label}",
        "model": "M1_tuned_SWLDA",
        "resampling_unit": "run_id; paired run resamples shared by both conditions",
        "bootstrap_replicates": n_boot,
        "seed": seed,
        "interval": "percentile 95% CI",
        "n_runs": n_runs,
        "n_epochs": int(len(meta_a["y"])),
        "n_characters": int(chars_a.sum()),
        "reference": {"label": reference_label, "scores": str(a_path), "scores_sha256": hash_a,
                      "epoch_auc": point_auc_a, "character_accuracy": point_acc_a},
        "candidate": {"label": candidate_label, "scores": str(b_path), "scores_sha256": hash_b,
                      "epoch_auc": point_auc_b, "character_accuracy": point_acc_b},
        "difference_candidate_minus_reference": {
            "epoch_auc": point_auc_b - point_auc_a,
            "epoch_auc_percentage_points": 100 * (point_auc_b - point_auc_a),
            "epoch_auc_ci95": np.quantile(delta_auc, [0.025, 0.975]).tolist(),
            "character_accuracy": point_acc_b - point_acc_a,
            "character_accuracy_percentage_points": 100 * (point_acc_b - point_acc_a),
            "character_accuracy_ci95": np.quantile(delta_acc, [0.025, 0.975]).tolist(),
        },
        "git_hash": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_dirty": bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT, text=True
        ).strip()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    args = parser.parse_args()
    if args.n_bootstrap < 1:
        parser.error("--n-bootstrap must be positive")
    print(json.dumps(run(args.reference, args.candidate, args.output,
                         args.n_bootstrap, args.seed,
                         args.reference_label, args.candidate_label), indent=2))


if __name__ == "__main__":
    main()
