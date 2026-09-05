"""
Mechanism validation sweep for the data-sufficiency gate.

For ONE real subject, holds real EEG evaluation sessions fixed and swaps
only the RAG phrase-bank file across increasing synthetic corpus sizes
(built by build_synthetic_validation_bank.py). Reports:

  - subject_gate_weight / sufficiency_gate_value at each tier (proves the
    gate ramps as designed, independent of accuracy)
  - accuracy / flashes-per-char / ITR at each tier (real numbers, real EEG)

This is a mechanism-validation experiment, kept separate from the main
Study D ablation table. Results go to results/tables/sufficiency_gate_sweep.json.

Usage:
    python3 scripts/sweep_sufficiency_gate.py --subject D_07 \
        --sizes 18 50 150 500 1000
"""
import argparse
import json
from pathlib import Path
import sys

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from run_pipeline import load_spelling_matrix, run_evaluation
from src.models.decoder import P300Decoder
from src.models.fusion import BayesianFusionEngine
from src.models.llm_predictor import LLMPredictor
from src.models.rag_predictor import RAGPredictor

GLOBAL_BANK_PATH = "data/rag/phrase_bank_global.csv"
SYNTH_DIR = Path("data/rag/synthetic_validation")
MODEL_PATH = Path("data/processed/swlda_model.pkl")
TEST_SESSIONS = Path("data/processed/test_sessions_by_subject.json")
RESULT_PATH = Path("results/tables/sufficiency_gate_sweep.json")
RAG_WEIGHT = 0.15
FUSION_ALPHA = 0.1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--sizes", nargs="+", type=int, required=True)
    args = parser.parse_args()

    with TEST_SESSIONS.open(encoding="utf-8") as f:
        test_sessions_by_subject = json.load(f)
    session_ids = test_sessions_by_subject[args.subject]

    spelling_matrix, n_rows, n_cols = load_spelling_matrix(
        "data/processed/grid_layout.json", study_name="StudyD"
    )
    num_classes = n_rows * n_cols
    clf = joblib.load(MODEL_PATH)

    results = []
    for size in sorted(args.sizes):
        bank_path = SYNTH_DIR / f"phrase_bank_{args.subject}_synth_{size}.csv"
        if not bank_path.exists():
            print(f"SKIP size={size}: {bank_path} not found "
                  f"(run build_synthetic_validation_bank.py first)")
            continue

        base_llm = LLMPredictor(spelling_matrix)
        predictor = RAGPredictor(
            base_llm,
            phrase_bank_path=GLOBAL_BANK_PATH,
            rag_weight=RAG_WEIGHT,
            retrieval_confidence_threshold=0.60,
            subject_id=args.subject,
            subject_only=True,
            min_subject_phrases=1,
        )
        # Point the predictor's subject bank loader at the synthetic file
        # for this tier instead of the real per-subject bank.
        predictor.subject_phrase_bank_path = bank_path
        predictor.subject_phrases = predictor._load_phrase_bank(bank_path)
        predictor.subject_token_count = sum(
            len(p.text.split()) for p in predictor.subject_phrases
        )

        fusion = BayesianFusionEngine(
            num_classes=num_classes, mode="fixed", base_alpha=FUSION_ALPHA
        )

        metrics, itr = run_evaluation(
            P300Decoder(spelling_matrix), predictor, fusion,
            n_rows=n_rows, n_cols=n_cols,
            flashes_per_seq=n_rows + n_cols, clf=clf,
            session_ids=session_ids, enable_session_growth=False,
        )
        flashes = metrics.total_flashes_used / max(1, metrics.total_characters)

        # Sample the gate's internal state directly for auditability --
        # re-run one retrieval_prior call on an arbitrary context so the
        # diagnostics reflect this tier's actual token count and gate value.
        _, diag = predictor.retrieval_prior("TH")

        row = {
            "corpus_size_words": size,
            "actual_token_count": predictor.subject_token_count,
            "sufficiency_gate_value": diag.sufficiency_gate_value,
            "subject_gate_weight_sample": diag.subject_gate_weight,
            "accuracy": metrics.accuracy,
            "flashes_per_character": flashes,
            "itr": itr,
        }
        results.append(row)
        print(f"size={size:5d}  sufficiency_gate={diag.sufficiency_gate_value:.3f}  "
              f"acc={metrics.accuracy:.2f}  flashes={flashes:.2f}  itr={itr:.2f}")

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(
        json.dumps({"subject": args.subject, "sweep": results}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {RESULT_PATH}")


if __name__ == "__main__":
    main()