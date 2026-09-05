"""Build global and per-subject RAG phrase banks from the session registry.

The train/test assignment is deterministic: each subject's sessions are split
with the SHA-256 rule used by the original global bank. No test target is
included in either the global or that subject's offline phrase bank.

Usage: python3 scripts/build_phrase_bank_from_registry.py
"""
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

REGISTRY = Path("data/processed/ground_truth_registry.json")
GLOBAL_BANK_OUT = Path("data/rag/phrase_bank_global.csv")
# Kept for compatibility with existing callers such as run_pipeline.py.
LEGACY_BANK_OUT = Path("data/rag/phrase_bank.csv")
SUBJECT_BANK_DIR = Path("data/rag/by_subject")
TEST_SPLIT_OUT = Path("data/processed/test_sessions.json")
TEST_BY_SUBJECT_OUT = Path("data/processed/test_sessions_by_subject.json")
SUBJECT_BANK_SIZES_OUT = Path("data/processed/subject_bank_sizes.json")
TEST_FRACTION = 0.2
DEFAULT_MIN_SUBJECT_PHRASES = 10
SUBJECT_ID_PATTERN = re.compile(r"^(?P<subject>.+?)_SE\d+(?:_|$)")


def subject_id_from_session_id(session_id: str) -> str:
    """Extract the subject portion preceding the first ``_SE<digits>``."""
    match = SUBJECT_ID_PATTERN.match(str(session_id))
    if match is None:
        raise ValueError(
            "Cannot extract subject ID from session ID "
            f"{session_id!r}; expected '<subject>_SE<digits>...'."
        )
    return match.group("subject")


def is_test(session_id: str) -> bool:
    h = int(hashlib.sha256(session_id.encode()).hexdigest(), 16)
    return (h % 100) < int(TEST_FRACTION * 100)


def write_phrase_bank(path: Path, words, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "source", "weight", "category"])
        for word in sorted(words):
            writer.writerow([word, source, 1.0, "corpus"])


def main():
    with REGISTRY.open(encoding="utf-8") as f:
        registry = json.load(f)

    subject_train_words = defaultdict(set)
    test_sessions_by_subject = defaultdict(list)
    global_train_words = set()
    for session_id, target_word in registry.items():
        subject_id = subject_id_from_session_id(session_id)
        if not isinstance(target_word, str) or not target_word.strip():
            continue
        target_word = target_word.strip().upper()
        if is_test(session_id):
            test_sessions_by_subject[subject_id].append(session_id)
        else:
            subject_train_words[subject_id].add(target_word)
            global_train_words.add(target_word)

    write_phrase_bank(GLOBAL_BANK_OUT, global_train_words, "studyd_train")
    write_phrase_bank(LEGACY_BANK_OUT, global_train_words, "studyd_train")

    all_subjects = sorted(set(subject_train_words) | set(test_sessions_by_subject))
    subject_bank_sizes = {}
    for subject_id in all_subjects:
        words = subject_train_words[subject_id]
        subject_bank_sizes[subject_id] = len(words)
        write_phrase_bank(
            SUBJECT_BANK_DIR / f"phrase_bank_{subject_id}.csv",
            words,
            "studyd_train_subject",
        )
        if len(words) < DEFAULT_MIN_SUBJECT_PHRASES:
            print(
                f"WARNING: {subject_id} has only {len(words)} unique train words "
                f"(< {DEFAULT_MIN_SUBJECT_PHRASES})."
            )

    test_sessions_by_subject = {
        subject_id: sorted(test_sessions_by_subject[subject_id])
        for subject_id in all_subjects
    }
    all_test_sessions = sorted(
        session_id
        for sessions in test_sessions_by_subject.values()
        for session_id in sessions
    )

    TEST_SPLIT_OUT.parent.mkdir(parents=True, exist_ok=True)
    with TEST_SPLIT_OUT.open("w", encoding="utf-8") as f:
        json.dump(all_test_sessions, f, indent=2)
    with TEST_BY_SUBJECT_OUT.open("w", encoding="utf-8") as f:
        json.dump(test_sessions_by_subject, f, indent=2, sort_keys=True)
    with SUBJECT_BANK_SIZES_OUT.open("w", encoding="utf-8") as f:
        json.dump(subject_bank_sizes, f, indent=2, sort_keys=True)

    print(f"global train unique targets: {len(global_train_words)} -> {GLOBAL_BANK_OUT}")
    print(f"test sessions: {len(all_test_sessions)} -> {TEST_SPLIT_OUT}")
    print(f"subject test splits -> {TEST_BY_SUBJECT_OUT}")
    print(f"subject bank sizes -> {SUBJECT_BANK_SIZES_OUT}")


if __name__ == "__main__":
    main()
