#!/usr/bin/env python3
"""Paired subject statistics and hierarchical character bootstrap for OOF results."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
from scipy.stats import rankdata, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _subjects_for_model(model_result: dict) -> dict[str, tuple[int, int]]:
    char_metrics = model_result["metrics"]["character_metrics"]
    output = {}
    for subject, values in char_metrics["per_subject"].items():
        n = int(values["characters"])
        correct = int(round(float(values["full_repetition_accuracy"]) * n))
        output[subject] = (correct, n)
    return output


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences != 0
    d = differences[nonzero]
    if not len(d):
        return 0.0
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def _bootstrap_delta(base: dict, candidate: dict, subjects: list[str], seed: int,
                     iterations: int = 20000) -> list[float]:
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for b in range(iterations):
        selected = rng.choice(subjects, size=len(subjects), replace=True)
        subject_deltas = []
        for subject in selected:
            base_correct, n = base[subject]
            cand_correct, _ = candidate[subject]
            # Resample character outcomes within subject, then subjects.
            base_rate = rng.binomial(n, base_correct / n) / n if n else 0.0
            cand_rate = rng.binomial(n, cand_correct / n) / n if n else 0.0
            subject_deltas.append(cand_rate - base_rate)
        samples[b] = np.mean(subject_deltas)
    return np.quantile(samples, [0.025, 0.975]).tolist()


def main() -> None:
    classical = json.loads((ROOT / "results/tables/classifier_classical_inner.json").read_text())
    screen = json.loads((ROOT / "results/tables/classifier_phase2_screens.json").read_text())
    comparisons = []
    for config, payload in classical.items():
        baseline = _subjects_for_model(payload["models"]["M0_clean_SGD"])
        candidates = {
            "M1_tuned_SWLDA": _subjects_for_model(payload["models"]["M1_tuned_SWLDA"]),
            "M2_shrinkage_LDA": {
                subject: (
                    int(round(values["full_repetition_accuracy"] * values["characters"])),
                    int(values["characters"]),
                )
                for subject, values in screen[
                    "baseline_off" if config == "baseline_off" else "bp_1_12"
                ]["character_metrics"]["per_subject"].items()
            },
        }
        for model_name, candidate in candidates.items():
            subjects = sorted(set(baseline) & set(candidate))
            base_rates = np.array([baseline[s][0] / baseline[s][1] for s in subjects])
            cand_rates = np.array([candidate[s][0] / candidate[s][1] for s in subjects])
            differences = cand_rates - base_rates
            try:
                p_value = float(wilcoxon(cand_rates, base_rates, alternative="two-sided").pvalue)
            except ValueError:
                p_value = 1.0
            ci = _bootstrap_delta(baseline, candidate, subjects, seed=407)
            comparisons.append({
                "config": config,
                "comparison": f"{model_name} vs M0_clean_SGD",
                "subjects": len(subjects),
                "mean_paired_difference": float(np.mean(differences)),
                "wilcoxon_p": p_value,
                "paired_rank_biserial": _rank_biserial(differences),
                "bootstrap_95_ci": ci,
            })

    # Holm correction across every model/config comparison.
    order = np.argsort([entry["wilcoxon_p"] for entry in comparisons])
    m = len(order)
    adjusted = np.ones(m, dtype=np.float64)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * comparisons[idx]["wilcoxon_p"]))
        adjusted[idx] = running
    for idx, entry in enumerate(comparisons):
        entry["holm_p"] = float(adjusted[idx])

    output = {
        "method": "paired Wilcoxon across subjects; Holm correction across all comparisons; hierarchical character bootstrap resampled within subjects",
        "seed": 407,
        "iterations": 20000,
        "comparisons": comparisons,
    }
    path = ROOT / "results/tables/classifier_classical_stats.json"
    path.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
