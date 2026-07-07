"""
Reconstructs the full ground-truth spelled sequence across an entire
session by processing every Train*.edf file in order (Train01, Train02,
...) and concatenating the single character each file yields, per the
discovery that one file = one character trial (repeated across many
sequences for a robust averaged ERP), not one file = one full phrase.

Also stacks all epochs + labels across files into one combined array,
ready for Day 3 classifier training.

Usage:
    python -m src.preprocessing.build_session_sequence /path/to/.../Train/CB
"""

import re
import sys
from pathlib import Path

import numpy as np

from src.preprocessing.epoching import parse_bigp3bci_studyb


def natural_sort_key(path: Path):
    match = re.search(r"(\d+)\.edf$", path.name)
    return int(match.group(1)) if match else 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m src.preprocessing.build_session_sequence /path/to/Train/CB")
        sys.exit(1)

    session_dir = Path(sys.argv[1]).expanduser().resolve()
    edf_files = sorted(session_dir.glob("*.edf"), key=natural_sort_key)

    if not edf_files:
        print(f"No .edf files found in {session_dir}")
        sys.exit(1)

    print(f"Found {len(edf_files)} files, processing in order...\n")

    all_X, all_y, spelled_chars, low_agreement_files = [], [], [], []

    for f in edf_files:
        print(f"--- {f.name} ---")
        epochs, char, _ = parse_bigp3bci_studyb(str(f))
        all_X.append(epochs.get_data())
        all_y.append(epochs.events[:, 2])
        spelled_chars.append(char)
        print()

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    full_sequence = "".join(spelled_chars)

    print("=" * 70)
    print(f"Reconstructed spelled sequence across session: '{full_sequence}'")
    print(f"Combined epochs shape: {X.shape}")
    print(f"Combined label distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    print("=" * 70)
    print("Sanity check: does the reconstructed sequence look like plausible")
    print("text (a real word/phrase), or garbled? If garbled, the per-file")
    print("majority-vote agreement fraction printed above for each file will")
    print("point to which specific file(s) are ambiguous.")

    return X, y, full_sequence


if __name__ == "__main__":
    main()