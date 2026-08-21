"""Sweep RAGPredictor hyperparameters on the held-out test split.

Usage: python scripts/sweep_rag_params.py
Requires run_pipeline.py's load_spelling_matrix / run_evaluation / decoder
setup already working (uses the same real SWLDA classifier + EEG data).
"""
import json
import os
import joblib

from src.models.decoder import P300Decoder
from src.models.fusion import BayesianFusionEngine
from src.models.llm_predictor import LLMPredictor
from src.models.rag_predictor import RAGPredictor
from run_pipeline import load_spelling_matrix, run_evaluation

WEIGHTS = [0.10, 0.15, 0.25, 0.35, 0.50]
THRESHOLDS = [0.0, 0.3, 0.6]

if __name__ == "__main__":
    spelling_matrix, n_rows, n_cols = load_spelling_matrix(
        "data/processed/grid_layout.json", study_name="StudyD"
    )
    num_classes = n_rows * n_cols
    flashes_per_seq = n_rows + n_cols

    decoder = P300Decoder(spelling_matrix)
    llm = LLMPredictor(spelling_matrix)

    clf = None
    MODEL_PATH = "data/processed/swlda_model.pkl"
    if os.path.exists(MODEL_PATH):
        clf = joblib.load(MODEL_PATH)

    eval_kwargs = dict(n_rows=n_rows, n_cols=n_cols,
                        flashes_per_seq=flashes_per_seq, clf=clf)

    results = []
    for w in WEIGHTS:
        for t in THRESHOLDS:
            rag_llm = RAGPredictor(
                llm, phrase_bank_path="data/rag/phrase_bank.csv",
                rag_weight=w, retrieval_confidence_threshold=t,
            )
            fusion = BayesianFusionEngine(num_classes=num_classes, mode='fixed', base_alpha=0.1)
            metrics, itr = run_evaluation(decoder, rag_llm, fusion, **eval_kwargs)
            fpc = metrics.total_flashes_used / max(1, metrics.total_characters)
            row = dict(rag_weight=w, threshold=t, accuracy=metrics.accuracy,
                       flashes_per_char=fpc, itr=itr)
            results.append(row)
            print(f"w={w:.2f} thr={t:.2f} -> acc={metrics.accuracy:.2f} "
                  f"fpc={fpc:.2f} itr={itr:.2f}")

    with open("results/tables/rag_sweep.json", "w") as f:
        json.dump(results, f, indent=2)

    best = max(results, key=lambda r: r["itr"])
    print("\nBEST by ITR:", best)