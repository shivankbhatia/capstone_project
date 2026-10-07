#!/usr/bin/env python3
"""Paired subject statistics for nested OOF fusion comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def _test_delta(values_a, values_b, rng, samples=20000):
    a, b = np.asarray(values_a, float), np.asarray(values_b, float)
    delta = a - b
    try:
        stat, p_value = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        stat, p_value = 0.0, 1.0
    boot = np.mean(rng.choice(delta, (samples, len(delta)), replace=True), axis=1)
    return {
        "subjects": len(delta),
        "mean_paired_delta": float(delta.mean()),
        "wilcoxon_statistic": float(stat),
        "p_uncorrected": float(p_value),
        "bootstrap_95_ci_subject_resampling": [
            float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))
        ],
    }


def _holm(p_values):
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    n = len(p_values)
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted, adjusted < 0.05


def run(input_path: Path, nested_path: Path, output_path: Path, seed=42):
    data = json.loads(input_path.read_text(encoding="utf-8"))
    nested = json.loads(nested_path.read_text(encoding="utf-8"))
    models = data["models"]
    if "M0" not in models:
        raise ValueError("Nested OOF fusion results must include M0")
    rng = np.random.default_rng(seed)
    comparisons = []
    base_auc = nested["M0"]["per_subject_raw_epoch_auc"]
    for model_name in models:
        if model_name == "M0" or model_name not in nested:
            continue
        candidate_auc = nested[model_name]["per_subject_raw_epoch_auc"]
        subjects = sorted(set(base_auc) & set(candidate_auc))
        comparisons.append({
            "comparison": f"{model_name} vs M0", "mode": "OOF epoch", "metric": "AUC",
            **_test_delta([candidate_auc[s] for s in subjects],
                          [base_auc[s] for s in subjects], rng),
        })
    for mode in ("eeg_only", "fusion"):
        for model_name in models:
            if model_name == "M0":
                continue
            reference_name = "M3a" if model_name == "M2_M3a_stack" else "M0"
            if reference_name not in models:
                continue
            baseline = models[reference_name][mode]["per_subject"]
            candidate = models[model_name][mode]["per_subject"]
            subjects = sorted(set(baseline) & set(candidate))
            for metric in ("stopped_accuracy", "sequence_row_column_nll"):
                if metric not in baseline[subjects[0]] or metric not in candidate[subjects[0]]:
                    continue
                comparisons.append({
                    "comparison": f"{model_name} vs {reference_name}",
                    "mode": mode,
                    "metric": metric,
                    **_test_delta(
                        [candidate[s][metric] for s in subjects],
                        [baseline[s][metric] for s in subjects], rng,
                    ),
                })
    p_adjusted, reject = _holm([c["p_uncorrected"] for c in comparisons])
    for row, p_adj, is_reject in zip(comparisons, p_adjusted, reject):
        row["p_holm"] = float(p_adj)
        row["significant_holm_0p05"] = bool(is_reject)
    result = {"source": str(input_path), "seed": seed, "correction": "Holm across all comparisons", "comparisons": comparisons}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--nested", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    result = run(args.input, args.nested, args.output, args.seed)
    print(json.dumps(result["comparisons"], indent=2))


if __name__ == "__main__":
    main()
