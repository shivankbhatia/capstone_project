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
    adjusted_p_value: float = 1.0
    holm_significant: bool = False


PERSONALIZATION_RUNGS = {
    "rung_0_classifier": "Classifier only",
    "rung_1_lm": "Classifier + plain LM fusion",
    "rung_2_global_rag": "Classifier + global pooled RAG",
    "rung_3_subject_only_rag": "Classifier + hard subject-only RAG",
    "rung_4_personalized_rag": (
        "Classifier + ramped subject/global RAG with session growth"
    ),
}


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

    def get_complete_subjects(
        self,
        required_methods=None,
    ) -> List[str]:
        """
        Return subjects for which all three experimental conditions exist.
        """

        if required_methods is None:
            required_methods = {"baseline", "fixed", "adaptive"}
        else:
            required_methods = set(required_methods)

        complete_subjects = []

        for subject_id, results in self.subject_results.items():
            if required_methods.issubset(results.keys()):
                complete_subjects.append(subject_id)

        return sorted(complete_subjects)

    def get_metric_arrays(
        self,
        metric: str,
        methods=None,
    ) -> Dict[str, np.ndarray]:
        """
        Return paired subject-level arrays.
        """

        methods = list(methods or ["baseline", "fixed", "adaptive"])
        subjects = self.get_complete_subjects(methods)

        if not subjects:
            raise ValueError(
                f"No subjects have complete results for {methods}."
            )

        return {
            method: np.array(
                [
                    self.subject_results[subject_id][method][metric]
                    for subject_id in subjects
                ],
                dtype=float,
            )
            for method in methods
        }

    @staticmethod
    def _apply_holm_bonferroni(
        results: List[SignificanceResult],
        alpha: float,
    ) -> List[SignificanceResult]:
        """Apply Holm's step-down correction across one result family."""
        ordered = sorted(enumerate(results), key=lambda item: item[1].p_value)
        running_adjusted = 0.0
        reject = True
        total = len(results)
        for rank, (index, result) in enumerate(ordered):
            adjusted = min(1.0, (total - rank) * result.p_value)
            running_adjusted = max(running_adjusted, adjusted)
            if result.p_value > alpha / (total - rank):
                reject = False
            result.adjusted_p_value = running_adjusted
            result.holm_significant = reject
            result.significant = reject
        return results

    def run_personalization_ladder_tests(
        self,
        alpha: float = 0.05,
    ) -> List[SignificanceResult]:
        """Test every rung pair per metric with one Holm-corrected family.

        Results remain paired at the subject level.  The returned collection
        includes the planned 2-vs-3 and 3-vs-4 comparisons as well as all
        other rung pairs, so correction covers the full comparison family.
        """
        methods = list(PERSONALIZATION_RUNGS)
        metrics = ["accuracy", "flashes_per_character", "itr"]
        results = []
        for metric in metrics:
            arrays = self.get_metric_arrays(metric, methods=methods)
            for baseline_index, baseline_method in enumerate(methods[:-1]):
                for comparison_method in methods[baseline_index + 1:]:
                    results.append(
                        _paired_wilcoxon(
                            baseline=arrays[baseline_method],
                            comparison=arrays[comparison_method],
                            comparison_name=(
                                f"{comparison_method} "
                                f"({PERSONALIZATION_RUNGS[comparison_method]})"
                            ),
                            metric_name=metric,
                            alpha=alpha,
                        )
                    )
        return self._apply_holm_bonferroni(results, alpha)

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
