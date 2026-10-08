#!/usr/bin/env python3
"""Summarize paired subject-level Q nested OOF fusion effects."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def summarize(source: Path, destination: Path, *, seed: int = 104729,
              bootstrap_samples: int = 20000) -> dict:
    result = json.loads(source.read_text(encoding="utf-8"))
    subjects = sorted(result["per_subject"])
    eeg = np.asarray([
        result["per_subject"][s]["eeg_only"]["correct"] /
        result["per_subject"][s]["eeg_only"]["n"] for s in subjects
    ])
    fusion = np.asarray([
        result["per_subject"][s]["lm_fusion"]["correct"] /
        result["per_subject"][s]["lm_fusion"]["n"] for s in subjects
    ])
    delta = fusion - eeg
    test = wilcoxon(fusion, eeg, alternative="two-sided", method="auto")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(subjects), size=(bootstrap_samples, len(subjects)))
    means = delta[draws].mean(axis=1)
    output = {
        "study": "StudyQ", "source": str(source.relative_to(ROOT)),
        "n_subjects": len(subjects), "paired_test": "two-sided Wilcoxon signed-rank",
        "wilcoxon_statistic": float(test.statistic), "p_value": float(test.pvalue),
        "mean_subject_accuracy_difference_fusion_minus_eeg": float(delta.mean()),
        "median_subject_accuracy_difference": float(np.median(delta)),
        "subjects_improved": int(np.sum(delta > 0)),
        "subjects_tied": int(np.sum(delta == 0)),
        "subjects_worse": int(np.sum(delta < 0)),
        "bootstrap_subject_resampling": {
            "seed": seed, "samples": bootstrap_samples,
            "mean_difference_ci_95": [float(x) for x in np.quantile(means, [0.025, 0.975])],
        },
        "per_subject": {
            s: {"eeg_only_accuracy": float(eeg[i]), "fusion_accuracy": float(fusion[i]),
                "difference": float(delta[i])} for i, s in enumerate(subjects)
        },
        "caveat": "Training-only nested OOF; not held-out manifest evaluation.",
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "results/tables/study_q_nested_fusion.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/study_q_nested_fusion_stats.json")
    args = parser.parse_args()
    print(json.dumps(summarize(args.input, args.output), indent=2))
