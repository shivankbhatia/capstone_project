"""Run the five-rung RAG-personalization ablation on held-out Study-D data.

The registry builder must run first, because this script consumes its global
and per-subject held-out session splits.  Results are aggregated per subject
and tested with paired, Holm-Bonferroni-corrected Wilcoxon comparisons.
"""
import argparse
import json
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_pipeline import load_spelling_matrix, run_evaluation
from src.evaluation.ablations import AblationTracker, PERSONALIZATION_RUNGS
from src.models.decoder import P300Decoder
from src.models.fusion import BayesianFusionEngine
from src.models.llm_predictor import LLMPredictor
from src.models.rag_predictor import RAGPredictor


TEST_SESSIONS = Path("splits/heldout_manifest.json")
MODEL_PATH = Path("data/processed/swlda_model.pkl")
GLOBAL_BANK_PATH = "data/rag/phrase_bank_global.csv"
RESULT_PATH = Path("results/tables/personalization_ablation_studyd.json")
RAG_WEIGHT = 0.15  # The existing held-out global-RAG configuration.
FUSION_ALPHA = 0.1


def build_condition(
    method, base_llm, num_classes, subject_id, disable_bigram_backoff=False,
    bigram_min_count=2, bigram_max_normalized_entropy=0.65, debug_trace_calls=0,
):
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
            rung_id=method, debug_trace_calls=debug_trace_calls,
        )
    elif method == "rung_3_subject_only_rag":
        predictor = RAGPredictor(
            base_llm, phrase_bank_path=GLOBAL_BANK_PATH,
            rag_weight=RAG_WEIGHT, retrieval_confidence_threshold=0.60,
            subject_id=subject_id, min_subject_phrases=1, subject_only=True,
            rung_id=method, disable_bigram_backoff=disable_bigram_backoff,
            bigram_min_count=bigram_min_count,
            bigram_max_normalized_entropy=bigram_max_normalized_entropy,
            debug_trace_calls=debug_trace_calls,
        )
    elif method == "rung_4_personalized_rag":
        predictor = RAGPredictor(
            base_llm, phrase_bank_path=GLOBAL_BANK_PATH,
            rag_weight=RAG_WEIGHT, retrieval_confidence_threshold=0.60,
            subject_id=subject_id,
            rung_id=method, disable_bigram_backoff=disable_bigram_backoff,
            bigram_min_count=bigram_min_count,
            bigram_max_normalized_entropy=bigram_max_normalized_entropy,
            debug_trace_calls=debug_trace_calls,
        )
    else:
        raise ValueError(f"Unknown ablation method: {method}")
    return predictor, BayesianFusionEngine(
        num_classes=num_classes, mode="fixed", base_alpha=FUSION_ALPHA
    ), method == "rung_4_personalized_rag"


def _blend_diagnostics(predictor):
    """Return per-character weights and backoff counts for auditability."""
    history = getattr(predictor, "blend_history", [])
    weights = [entry["blend_weight"] for entry in history]
    return {
        "char_backoff_count": getattr(predictor, "char_backoff_count", 0),
        "n_character_decisions": len(history),
        "blend_weight_min": min(weights, default=0.0),
        "blend_weight_max": max(weights, default=0.0),
        "blend_weight_mean": sum(weights) / len(weights) if weights else 0.0,
        "source_counts": getattr(predictor, "source_counts", {}),
        "flip_counts": getattr(predictor, "flip_counts", {}),
        "flip_rate": (
            getattr(predictor, "flip_counts", {}).get("total", 0) / len(history)
            if history else 0.0
        ),
        "character_decisions": history,
    }


def main():
    global RESULT_PATH
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug-subject", default="D_01",
        help="Subject whose Rung 3/4 per-character blend weights are saved.",
    )
    parser.add_argument(
        "--subject", action="append",
        help="Evaluate only this subject; repeat to run a resumable batch.",
    )
    parser.add_argument(
        "--reset-results", action="store_true",
        help="Discard an existing result table before a resumable run.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Rerun already checkpointed rungs for selected subjects.",
    )
    parser.add_argument(
        "--disable-bigram-backoff", action="store_true",
        help="Use the base prior when no subject word-level continuation exists.",
    )
    parser.add_argument("--bigram-min-count", type=int, default=2)
    parser.add_argument("--bigram-max-normalized-entropy", type=float, default=0.65)
    parser.add_argument(
        "--allow-heldout-eval", action="store_true",
        help="Required to read sealed Study D evaluation labels for the final ladder replay.",
    )
    parser.add_argument("--tau", type=float, default=0.80)
    parser.add_argument("--min-sequences", type=int, default=2)
    parser.add_argument("--max-sequences", type=int, default=10)
    parser.add_argument(
        "--debug-trace-calls", type=int, default=0,
        help="Print the first N RAG distributions per rung.",
    )
    parser.add_argument(
        "--output", type=Path,
        help="Optional result-table path for an isolated diagnostic run.",
    )
    args = parser.parse_args()
    if not args.allow_heldout_eval:
        parser.error("pass --allow-heldout-eval only for the pre-registered final evaluation")
    if args.output:
        RESULT_PATH = args.output
    with TEST_SESSIONS.open(encoding="utf-8") as handle:
        test_sessions_by_subject = json.load(handle)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Required classifier is missing: {MODEL_PATH}")

    spelling_matrix, n_rows, n_cols = load_spelling_matrix(
        "data/processed/grid_layout.json", study_name="StudyD"
    )
    num_classes = n_rows * n_cols
    clf = joblib.load(MODEL_PATH)
    existing = {}
    if RESULT_PATH.exists() and not args.reset_results:
        existing = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    raw_results = existing.get("subjects", {})
    tracker = AblationTracker()
    debug_diagnostics = existing.get("blend_diagnostics", {}).get(
        args.debug_subject, {}
    )
    all_subject_diagnostics = existing.get("subject_diagnostics", {})
    selected_subjects = set(args.subject or test_sessions_by_subject)
    unknown_subjects = selected_subjects.difference(test_sessions_by_subject)
    if unknown_subjects:
        raise ValueError(f"Unknown subject(s): {sorted(unknown_subjects)}")

    for subject_id, session_ids in sorted(test_sessions_by_subject.items()):
        if subject_id not in selected_subjects:
            continue
        raw_results.setdefault(subject_id, {})
        for method in PERSONALIZATION_RUNGS:
            # A full five-rung evaluation can exceed an interactive job's
            # time limit.  Keep completed rungs and resume safely next run.
            if method in raw_results[subject_id] and not args.force:
                continue
            base_llm = LLMPredictor(spelling_matrix)
            predictor, fusion, growth = build_condition(
                method, base_llm, num_classes, subject_id,
                disable_bigram_backoff=args.disable_bigram_backoff,
                bigram_min_count=args.bigram_min_count,
                bigram_max_normalized_entropy=args.bigram_max_normalized_entropy,
                debug_trace_calls=args.debug_trace_calls,
            )
            metrics, itr = run_evaluation(
                P300Decoder(spelling_matrix), predictor, fusion,
                n_rows=n_rows, n_cols=n_cols,
                flashes_per_seq=n_rows + n_cols, clf=clf,
                session_ids=session_ids, enable_session_growth=growth,
                allow_heldout_labels=True,
                confidence_threshold=args.tau,
                min_flashes=args.min_sequences,
                max_sequences=args.max_sequences,
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
            RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
            RESULT_PATH.write_text(
                json.dumps({
                    "subjects": raw_results,
                    "significance": [],
                    "blend_diagnostics": {args.debug_subject: debug_diagnostics},
                    "subject_diagnostics": all_subject_diagnostics,
                }, indent=2),
                encoding="utf-8",
            )
            print(
                f"{subject_id} {method}: accuracy={metrics.accuracy:.2f}, "
                f"flashes={flashes:.2f}, ITR={itr:.2f}"
            )
            if method in {
                "rung_3_subject_only_rag", "rung_4_personalized_rag"
            }:
                diagnostic = _blend_diagnostics(predictor)
                all_subject_diagnostics.setdefault(subject_id, {})[method] = diagnostic
                if subject_id == args.debug_subject:
                    debug_diagnostics[method] = diagnostic
                    print(
                        f"  blend diagnostics: n={diagnostic['n_character_decisions']}, "
                        f"range=[{diagnostic['blend_weight_min']:.6f}, "
                        f"{diagnostic['blend_weight_max']:.6f}], "
                        f"backoff={diagnostic['char_backoff_count']}"
                    )

    complete = set(test_sessions_by_subject).issubset(raw_results)
    if complete:
        tracker = AblationTracker()
        for subject_id, methods in raw_results.items():
            for method, values in methods.items():
                tracker.add_subject_result(
                    subject_id, method, values["accuracy"],
                    values["flashes_per_character"], values["itr"],
                )
        significance = [
            result.__dict__ for result in tracker.run_personalization_ladder_tests()
        ]
    else:
        significance = []
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps({
            "subjects": raw_results,
            "significance": significance,
            "blend_diagnostics": {args.debug_subject: debug_diagnostics},
            "subject_diagnostics": all_subject_diagnostics,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
