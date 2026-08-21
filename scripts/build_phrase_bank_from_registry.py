"""Build data/rag/phrase_bank.csv from ground_truth_registry.json.

Splits sessions 80/20 (train/test) via deterministic hash of session_id.
Phrase bank is built ONLY from train sessions. Writes data/processed/test_sessions.json
so run_pipeline.py can be restricted to eval on the held-out test split.

Usage: python scripts/build_phrase_bank_from_registry.py
"""
import csv
import hashlib
import json
import os

REGISTRY = "data/processed/ground_truth_registry.json"
BANK_OUT = "data/rag/phrase_bank.csv"
TEST_SPLIT_OUT = "data/processed/test_sessions.json"
TEST_FRACTION = 0.2


def is_test(session_id: str) -> bool:
    h = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
    return (h % 100) < int(TEST_FRACTION * 100)


def main():
    with open(REGISTRY) as f:
        registry = json.load(f)

    train_words, test_sessions = set(), []
    for session_id, target_word in registry.items():
        if not isinstance(target_word, str) or not target_word.strip():
            continue
        if is_test(session_id):
            test_sessions.append(session_id)
        else:
            train_words.add(target_word.strip().upper())

    os.makedirs(os.path.dirname(BANK_OUT), exist_ok=True)
    with open(BANK_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "source", "weight", "category"])
        for word in sorted(train_words):
            w.writerow([word, "studyd_train", 1.0, "corpus"])

    with open(TEST_SPLIT_OUT, "w") as f:
        json.dump(sorted(test_sessions), f, indent=2)

    print(f"train unique targets: {len(train_words)} -> {BANK_OUT}")
    print(f"test sessions: {len(test_sessions)} -> {TEST_SPLIT_OUT}")


if __name__ == "__main__":
    main()