"""Check held-out test session/trial counts per subject to pick a higher-n
subject (or set of subjects) for the sufficiency-gate sweep."""
import json
from pathlib import Path

TEST_SESSIONS = Path("data/processed/test_sessions_by_subject.json")

def main():
    if not TEST_SESSIONS.exists():
        print(f"MISSING: {TEST_SESSIONS}")
        return
    with TEST_SESSIONS.open(encoding="utf-8") as f:
        d = json.load(f)
    print(f"{'Subject':10s} {'#Sessions':>10s}")
    for subject, sessions in sorted(d.items()):
        print(f"{subject:10s} {len(sessions):10d}")
    total = sum(len(v) for v in d.values())
    print(f"\nTotal held-out sessions across all subjects: {total}")

if __name__ == "__main__":
    main()