"""Run the five-rung RAG-personalization ablation on held-out Study-D data.

The registry builder must run first, because this script consumes its global
and per-subject held-out session splits.  Results are aggregated per subject
and tested with paired, Holm-Bonferroni-corrected Wilcoxon comparisons.
"""
import json
from pathlib import Path

import joblib

from run_pipeline import load_spelling_matrix, run_evaluation
from src.evaluation.ablations import AblationTracker, PERSONALIZATION_RUNGS
from src.models.decoder import P300Decoder
from src.models.fusion import BayesianFusionEngine
from src.models.llm_predictor import LLMPredictor
from src.models.rag_predictor import RAGPredictor


TEST_SESSIONS = Path("data/processed/test_sessions_by_subject.json")
MODEL_PATH = Path("data/processed/swlda_model.pkl")
GLOBAL_BANK_PATH = "data/rag/phrase_bank_global.csv"
RESULT_PATH = Path("results/tables/personalization_ablation_studyd.json")
RAG_WEIGHT = 0.15  # The existing held-out global-RAG configuration.
FUSION_ALPHA = 0.1


def build_condition(method, base_llm, num_classes, subject_id):
    if method == "rung_0_classifier":
        fusion = BayesianFusionEngine(
            num_classes=num_classes, mode="fixed", base_alpha=0.0
        )
        fusion.toggle(False)
        return base_llm, fusion, False
    if method == "rung_1_lm":
        return base_llm, BayesianFusionEngine(
            num_classes=num_classes, mode="fixed", base_alpha=FUSION_ALPHA
        ), False
    if method == "rung_2_global_rag":
        predictor = RAGPredictor(
            base_llm, phrase_bank_path=GLOBAL_BANK_PATH,
            rag_weight=RAG_WEIGHT, retrieval_confidence_threshold=0.60,
        )
    elif method == "rung_3_subject_only_rag":
        predictor = RAGPredictor(
            base_llm, phrase_bank_path=GLOBAL_BANK_PATH,
            rag_weight=RAG_WEIGHT, retrieval_confidence_threshold=0.60,
            subject_id=subject_id, min_subject_phrases=1, subject_only=True,
        )
    elif method == "rung_4_personalized_rag":
        predictor = RAGPredictor(
            base_llm, phrase_bank_path=GLOBAL_BANK_PATH,
            rag_weight=RAG_WEIGHT, retrieval_confidence_threshold=0.60,
            subject_id=subject_id,
        )
    else:
        raise ValueError(f"Unknown ablation method: {method}")
    return predictor, BayesianFusionEngine(
        num_classes=num_classes, mode="fixed", base_alpha=FUSION_ALPHA
    ), method == "rung_4_personalized_rag"


def main():
    with TEST_SESSIONS.open(encoding="utf-8") as handle:
        test_sessions_by_subject = json.load(handle)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Required classifier is missing: {MODEL_PATH}")

    spelling_matrix, n_rows, n_cols = load_spelling_matrix(
        "data/processed/grid_layout.json", study_name="StudyD"
    )
    num_classes = n_rows * n_cols
    clf = joblib.load(MODEL_PATH)
    tracker = AblationTracker()
    raw_results = {}

    for subject_id, session_ids in sorted(test_sessions_by_subject.items()):
        raw_results[subject_id] = {}
        for method in PERSONALIZATION_RUNGS:
            base_llm = LLMPredictor(spelling_matrix)
            predictor, fusion, growth = build_condition(
                method, base_llm, num_classes, subject_id
            )
            metrics, itr = run_evaluation(
                P300Decoder(spelling_matrix), predictor, fusion,
                n_rows=n_rows, n_cols=n_cols,
                flashes_per_seq=n_rows + n_cols, clf=clf,
                session_ids=session_ids, enable_session_growth=growth,
            )
            flashes = metrics.total_flashes_used / max(1, metrics.total_characters)
            tracker.add_subject_result(
                subject_id, method, metrics.accuracy, flashes, itr
            )
            raw_results[subject_id][method] = {
                "accuracy": metrics.accuracy,
                "flashes_per_character": flashes,
                "itr": itr,
            }
            print(
                f"{subject_id} {method}: accuracy={metrics.accuracy:.2f}, "
                f"flashes={flashes:.2f}, ITR={itr:.2f}"
            )

    significance = [result.__dict__ for result in tracker.run_personalization_ladder_tests()]
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps({"subjects": raw_results, "significance": significance}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
