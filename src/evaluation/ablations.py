"""
Day 8 — Ablation runner
LM fusion on/off, RAG on/off, stopping-threshold sweep, cross-subject
comparison. Fill in during Day 8.
"""
from dataclasses import dataclass
from typing import Dict, List, Any

import numpy as np
from scipy.stats import wilcoxon


@dataclass
class SignificanceResult:
    comparison: str
    metric: str
    n_subjects: int
    statistic: float
    p_value: float
    mean_difference: float
    median_difference: float
    significant: bool


def _paired_wilcoxon(
    baseline: np.ndarray,
    comparison: np.ndarray,
    comparison_name: str,
    metric_name: str,
    alpha: float = 0.05,
) -> SignificanceResult:
    """
    Perform a paired Wilcoxon signed-rank test.

    Difference is defined as:

        comparison - baseline

    Therefore:
        positive mean_difference => comparison performed better
        for accuracy / ITR

    For flashes:
        negative mean_difference => comparison used fewer flashes
    """

    baseline = np.asarray(baseline, dtype=float)
    comparison = np.asarray(comparison, dtype=float)

    if len(baseline) != len(comparison):
        raise ValueError(
            f"Paired samples must have equal length: "
            f"{len(baseline)} vs {len(comparison)}"
        )

    if len(baseline) < 2:
        raise ValueError(
            "At least 2 paired subjects are required "
            "for significance testing."
        )

    differences = comparison - baseline

    # Wilcoxon cannot test a vector where every paired difference is zero.
    if np.allclose(differences, 0.0):
        statistic = 0.0
        p_value = 1.0
    else:
        result = wilcoxon(
            baseline,
            comparison,
            alternative="two-sided",
            zero_method="wilcox",
            method="auto",
        )

        statistic = float(result.statistic)
        p_value = float(result.pvalue)

    return SignificanceResult(
        comparison=f"Baseline vs {comparison_name}",
        metric=metric_name,
        n_subjects=len(baseline),
        statistic=statistic,
        p_value=p_value,
        mean_difference=float(np.mean(differences)),
        median_difference=float(np.median(differences)),
        significant=p_value < alpha,
    )


class AblationTracker:
    """
    Stores results at the per-subject level and performs paired
    statistical comparisons between experimental conditions.
    """

    def __init__(self):
        self.subject_results: Dict[str, Dict[str, Dict[str, float]]] = {}

    def add_subject_result(
        self,
        subject_id: str,
        method: str,
        accuracy: float,
        flashes_per_character: float,
        itr: float,
    ):
        """
        Store one aggregated result for a subject and decoding method.
        """

        if subject_id not in self.subject_results:
            self.subject_results[subject_id] = {}

        self.subject_results[subject_id][method] = {
            "accuracy": float(accuracy),
            "flashes_per_character": float(flashes_per_character),
            "itr": float(itr),
        }

    def get_complete_subjects(self) -> List[str]:
        """
        Return subjects for which all three experimental conditions exist.
        """

        required_methods = {
            "baseline",
            "fixed",
            "adaptive",
        }

        complete_subjects = []

        for subject_id, results in self.subject_results.items():
            if required_methods.issubset(results.keys()):
                complete_subjects.append(subject_id)

        return sorted(complete_subjects)

    def get_metric_arrays(
        self,
        metric: str,
    ) -> Dict[str, np.ndarray]:
        """
        Return paired subject-level arrays.
        """

        subjects = self.get_complete_subjects()

        if not subjects:
            raise ValueError(
                "No subjects have complete Baseline/Fixed/Adaptive results."
            )

        return {
            method: np.array(
                [
                    self.subject_results[subject_id][method][metric]
                    for subject_id in subjects
                ],
                dtype=float,
            )
            for method in ["baseline", "fixed", "adaptive"]
        }

    def run_significance_tests(
        self,
        alpha: float = 0.05,
    ) -> List[SignificanceResult]:
        """
        Run paired Wilcoxon signed-rank tests:

            Baseline vs Fixed
            Baseline vs Adaptive

        for:
            - Accuracy
            - Flashes per character
            - ITR
        """

        metrics = [
            "accuracy",
            "flashes_per_character",
            "itr",
        ]

        results = []

        for metric in metrics:
            arrays = self.get_metric_arrays(metric)

            results.append(
                _paired_wilcoxon(
                    baseline=arrays["baseline"],
                    comparison=arrays["fixed"],
                    comparison_name="Fixed (alpha=0.1)",
                    metric_name=metric,
                    alpha=alpha,
                )
            )

            results.append(
                _paired_wilcoxon(
                    baseline=arrays["baseline"],
                    comparison=arrays["adaptive"],
                    comparison_name="Adaptive",
                    metric_name=metric,
                    alpha=alpha,
                )
            )

        return results

    def print_significance_results(
        self,
        alpha: float = 0.05,
    ):
        results = self.run_significance_tests(alpha=alpha)

        print("\n" + "=" * 80)
        print("STATISTICAL SIGNIFICANCE TESTING")
        print("Paired Wilcoxon signed-rank tests")
        print("=" * 80)

        for result in results:
            print(f"\nComparison: {result.comparison}")
            print(f"Metric:     {result.metric}")
            print(f"Subjects:   {result.n_subjects}")
            print(f"Statistic:  {result.statistic:.4f}")
            print(f"P-value:    {result.p_value:.6f}")
            print(
                f"Mean Δ:     {result.mean_difference:+.6f}"
            )
            print(
                f"Median Δ:   {result.median_difference:+.6f}"
            )
            print(
                f"Significant at α={alpha}: "
                f"{'YES' if result.significant else 'NO'}"
            )

        return results