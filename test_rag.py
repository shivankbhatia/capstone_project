import numpy as np
from src.models.rag_predictor import RAGPredictor


class DummyPredictor:
    # Matches the structure expected by RAGPredictor
    char_list = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ ") + ["Sp"]

    def predict_next_char(self, context_so_far):
        # Uniform base prior
        return np.ones(len(self.char_list)) / len(self.char_list)


base = DummyPredictor()

rag = RAGPredictor(
    base_predictor=base,
    phrase_bank_path="data/rag/phrase_bank.csv",
    rag_weight=0.25,
    retrieval_confidence_threshold=0.60,
)

tests = [
    "",
    "HEL",
    "I NEED ",
    "TURN ON THE L",
    "BUT",
    "HID",
    "XYZ",
]

for context in tests:
    probs = rag.predict_next_char(context)
    top_indices = np.argsort(probs)[::-1][:5]

    print("\nContext:", repr(context))
    print("Top predictions:")

    for idx in top_indices:
        print(f"  {rag.char_list[idx]!r}: {probs[idx]:.4f}")

    print("Diagnostics:")
    print("  Partial word:", rag.last_diagnostics.partial_word)
    print("  Matched:", rag.last_diagnostics.matched_phrases)
    print("  Confidence:", rag.last_diagnostics.retrieval_confidence)
    print("  Enabled:", rag.last_diagnostics.rag_enabled)
    print("  Reason:", rag.last_diagnostics.reason)
