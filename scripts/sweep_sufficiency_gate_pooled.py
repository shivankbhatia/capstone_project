"""Pooled, leakage-checked mechanism sweep for the sufficiency gate.

Synthetic validation banks are used only from data/rag/synthetic_validation.
Raw decoding counts are pooled across subjects before accuracy, flashes, and
ITR are calculated; subject percentages are never averaged.
"""
import argparse
import json
import sys
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_pipeline import load_spelling_matrix, run_evaluation
from src.models.decoder import P300Decoder, calculate_itr
from src.models.fusion import BayesianFusionEngine
from src.models.llm_predictor import LLMPredictor
from src.models.rag_predictor import RAGPredictor

SUBJECTS = ["D_01", "D_03", "D_04", "D_05", "D_08", "D_10", "D_11", "D_13"]
SIZES = [18, 50, 150, 500, 1000]
SYNTH_DIR = Path("data/rag/synthetic_validation")
OUT = Path("results/tables/sufficiency_gate_pooled_sweep.json")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subjects", nargs="+", default=SUBJECTS)
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES)
    args = parser.parse_args()
    session_map = json.loads(Path("data/processed/test_sessions_by_subject.json").read_text())
    matrix, rows, cols = load_spelling_matrix("data/processed/grid_layout.json", "StudyD")
    clf = joblib.load("data/processed/swlda_model.pkl")
    output = []
    for size in sorted(args.sizes):
        totals = {"correct": 0, "characters": 0, "flashes": 0}
        gates = []
        for subject in args.subjects:
            bank = SYNTH_DIR / f"phrase_bank_{subject}_synth_{size}.csv"
            if not bank.exists():
                raise FileNotFoundError(bank)
            predictor = RAGPredictor(LLMPredictor(matrix), phrase_bank_path="data/rag/phrase_bank_global.csv", rag_weight=.15, subject_id=subject, subject_only=True, min_subject_phrases=1)
            predictor.subject_phrases = predictor._load_phrase_bank(bank)
            predictor.subject_token_count = sum(len(p.text.split()) for p in predictor.subject_phrases)
            fusion = BayesianFusionEngine(num_classes=rows * cols, mode="fixed", base_alpha=.1)
            metrics, _ = run_evaluation(P300Decoder(matrix), predictor, fusion, rows, cols, rows + cols, clf=clf, session_ids=session_map[subject])
            totals["correct"] += metrics.correct_characters
            totals["characters"] += metrics.total_characters
            totals["flashes"] += metrics.total_flashes_used
            gates.append(predictor.retrieval_prior("TH")[1].sufficiency_gate_value)
        accuracy = 100 * totals["correct"] / totals["characters"]
        flashes = totals["flashes"] / totals["characters"]
        output.append({"corpus_size_words": size, "subjects": args.subjects, "subject_count": len(args.subjects), "correct_characters": totals["correct"], "total_characters": totals["characters"], "total_flashes_used": totals["flashes"], "accuracy": accuracy, "flashes_per_character": flashes, "itr": calculate_itr(rows * cols, accuracy, flashes * 2.0), "mean_sufficiency_gate": sum(gates) / len(gates)})
        print(output[-1])
    OUT.write_text(json.dumps({"experiment": "synthetic_validation_only", "pooled_by_raw_counts": True, "sweep": output}, indent=2))

if __name__ == "__main__": main()
