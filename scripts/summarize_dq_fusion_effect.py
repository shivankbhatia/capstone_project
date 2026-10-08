#!/usr/bin/env python3
"""Pool paired subject-level M0 fusion effects from clean D and Q inner OOF."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp, ttest_ind, wilcoxon

ROOT = Path(__file__).resolve().parents[1]


def _delta_d(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    model = report["models"]["M0"]
    eeg, fusion = model["eeg_only"]["per_subject"], model["fusion"]["per_subject"]
    return {s: fusion[s]["stopped_accuracy"] - eeg[s]["stopped_accuracy"] for s in eeg}


def _delta_q(path: Path) -> dict[str, float]:
    report = json.loads(path.read_text(encoding="utf-8"))
    return {s: values["difference"] for s, values in report["per_subject"].items()}


def summarize(d_path: Path, q_path: Path, out: Path, *, seed: int = 271828,
              bootstrap_samples: int = 30000) -> dict:
    groups = {"D": _delta_d(d_path), "Q": _delta_q(q_path)}
    arrays = {name: np.asarray(list(values.values()), dtype=float) for name, values in groups.items()}
    all_delta = np.concatenate([arrays["D"], arrays["Q"]])
    # Differencing the two within-subject conditions removes the subject
    # intercept; this estimates a study-stratified fixed fusion effect.
    rng = np.random.default_rng(seed)
    d = arrays["D"][rng.integers(0, len(arrays["D"]), (bootstrap_samples, len(arrays["D"])))].mean(axis=1)
    q = arrays["Q"][rng.integers(0, len(arrays["Q"]), (bootstrap_samples, len(arrays["Q"])))].mean(axis=1)
    pooled = (17 * d + 36 * q) / (17 + 36)
    d_test = wilcoxon(arrays["D"], alternative="two-sided", method="auto")
    q_test = wilcoxon(arrays["Q"], alternative="two-sided", method="auto")
    raw_p = {"D": float(d_test.pvalue), "Q": float(q_test.pvalue)}
    ordered = sorted(raw_p, key=raw_p.get)
    holm = {ordered[0]: min(1.0, 2 * raw_p[ordered[0]]),
            ordered[1]: min(1.0, raw_p[ordered[1]])}
    overall_test = ttest_1samp(all_delta, 0.0)
    heterogeneity = ttest_ind(arrays["D"], arrays["Q"], equal_var=False)
    studies = {}
    for name, delta in arrays.items():
        studies[name] = {
            "n_subjects": int(len(delta)), "mean_delta": float(delta.mean()),
            "median_delta": float(np.median(delta)),
            "improved_tied_worse": [int(np.sum(delta > 0)), int(np.sum(delta == 0)), int(np.sum(delta < 0))],
            "wilcoxon_p_unadjusted": raw_p[name], "wilcoxon_p_holm_two_studies": holm[name],
        }
    output = {
        "endpoint": "subject-level difference in fixed-policy character accuracy (fusion minus EEG-only)",
        "model": "clean M0; fixed fusion alpha 0.1 and tau 0.8/min 2/max 10",
        "analysis": "paired within-subject differences with study-stratified fixed effect; subject intercept removed by differencing",
        "studies": studies,
        "pooled_subject_weighted": {
            "n_subjects": int(len(all_delta)), "mean_delta": float(all_delta.mean()),
            "median_delta": float(np.median(all_delta)),
            "improved_tied_worse": [int(np.sum(all_delta > 0)), int(np.sum(all_delta == 0)), int(np.sum(all_delta < 0))],
            "one_sample_t_p_unadjusted": float(overall_test.pvalue),
            "bootstrap_95_ci_mean_delta_stratified": [float(v) for v in np.quantile(pooled, [0.025, 0.975])],
        },
        "study_interaction": {
            "welch_t_p_unadjusted": float(heterogeneity.pvalue),
            "mean_effect_difference_Q_minus_D": float(arrays["Q"].mean() - arrays["D"].mean()),
        },
        "bootstrap": {"seed": seed, "samples": bootstrap_samples},
        "caveats": [
            "Inner OOF only; no manifest labels or predictions used.",
            "Q uses a group-flash decoder while D uses row/column flashes; interpret pooled effect as cross-study support, not exact measurement equivalence.",
            "The Q language prior is evaluated with teacher-forced previous target keys; this is not a prospective natural-language RAG result.",
        ],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--d", type=Path, default=ROOT / "results/tables/classifier_inner_fusion.json")
    parser.add_argument("--q", type=Path, default=ROOT / "results/tables/study_q_nested_fusion_stats.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/tables/dq_inner_fusion_effect.json")
    args = parser.parse_args()
    print(json.dumps(summarize(args.d, args.q, args.output), indent=2))
