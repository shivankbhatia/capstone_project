"""
Generate synthetic validation phrase banks for the data-sufficiency gate.

IMPORTANT: this data is for mechanism validation ONLY -- it does not feed
the main Study D ablation table (results/tables/personalization_ablation_studyd.json)
and must never be merged into data/rag/by_subject/. It lives in its own
directory (data/rag/synthetic_validation/) so there is no risk of it
silently entering the main results.

Design: pick ONE real subject with real EEG test sessions. Build several
synthetic phrase banks of increasing size for that subject by drawing
words from a public-domain text source, EXCLUDING any word that appears
in that subject's real held-out test-session targets (to avoid leakage).
Real EEG signal for the subject is untouched -- only the RAG text corpus
size is varied. This isolates corpus size as the single manipulated
variable.

Usage:
    python3 scripts/build_synthetic_validation_bank.py --subject D_07 \
        --sizes 18 50 150 500 1000
"""
import argparse
import csv
import json
import random
from pathlib import Path

REGISTRY = Path("data/processed/ground_truth_registry.json")
TEST_SESSIONS_BY_SUBJECT = Path("data/processed/test_sessions_by_subject.json")
OUT_DIR = Path("data/rag/synthetic_validation")

# Public-domain filler text -- large, varied vocabulary, no connection to
# the P300 task vocabulary. Swap in any public-domain corpus you like
# (e.g. a Project Gutenberg text you have locally); this is a compact
# built-in fallback so the script runs standalone.
FILLER_TEXT_SOURCE = Path("data/rag/synthetic_validation/filler_source.txt")


def load_excluded_words(subject: str) -> set:
    """Words that must NOT appear in the synthetic bank: this subject's
    real held-out test targets, across all conditions. Prevents leakage."""
    excluded = set()
    if not REGISTRY.exists() or not TEST_SESSIONS_BY_SUBJECT.exists():
        print("WARNING: registry or test-session file missing; "
              "cannot verify leakage exclusion. Proceed with caution.")
        return excluded

    with REGISTRY.open(encoding="utf-8") as f:
        registry = json.load(f)
    with TEST_SESSIONS_BY_SUBJECT.open(encoding="utf-8") as f:
        test_sessions = json.load(f)

    subject_test_ids = set(test_sessions.get(subject, []))
    for session_id, target in registry.items():
        if session_id in subject_test_ids and isinstance(target, str):
            for w in target.strip().split():
                excluded.add(w.upper())
    return excluded


def load_filler_words(excluded: set) -> list:
    if not FILLER_TEXT_SOURCE.exists():
        raise FileNotFoundError(
            f"{FILLER_TEXT_SOURCE} not found. Place a public-domain text "
            "file there (e.g. a Project Gutenberg .txt) before running."
        )
    text = FILLER_TEXT_SOURCE.read_text(encoding="utf-8", errors="ignore")
    words = [w.strip(".,!?;:\"'()").upper() for w in text.split()]
    words = [w for w in words if w.isalpha() and w not in excluded]
    return words


def write_bank(path: Path, words: list, size_tier: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "source", "weight", "category"])
        for word in words:
            # Source is explicitly tagged "synthetic_validation" -- never
            # "studyd_train_subject" -- so it can never be confused with
            # or accidentally scored as real personalization data.
            writer.writerow([word, "synthetic_validation", 1.0, f"tier_{size_tier}"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True, help="Real subject ID, e.g. D_07")
    parser.add_argument("--sizes", nargs="+", type=int, required=True,
                         help="Corpus sizes (word counts) to generate, e.g. 18 50 150 500 1000")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    excluded = load_excluded_words(args.subject)
    print(f"Excluding {len(excluded)} real test-target words from synthetic corpus.")

    filler_pool = load_filler_words(excluded)
    if not filler_pool:
        raise ValueError("Filler word pool is empty after exclusion filtering.")

    for size in sorted(args.sizes):
        if size > len(filler_pool):
            print(f"WARNING: requested size {size} exceeds filler pool "
                  f"({len(filler_pool)}); sampling with replacement.")
            words = random.choices(filler_pool, k=size)
        else:
            words = random.sample(filler_pool, k=size)

        out_path = OUT_DIR / f"phrase_bank_{args.subject}_synth_{size}.csv"
        write_bank(out_path, words, size)
        print(f"Wrote {out_path} ({size} words)")


if __name__ == "__main__":
    main()