#!/usr/bin/env python3
"""Build Q RAG banks from the dedicated clean Q training registry only."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "data/processed/study_q_train_registry.json"
CACHE = ROOT / "data/cache/study_q_classifier_epochs.h5"
LAYOUT = ROOT / "data/processed/grid_layout.json"
OUT = ROOT / "data/rag/study_q"


def _write_bank(path: Path, phrases: set[str], source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "source", "weight", "category"))
        writer.writeheader()
        for phrase in sorted(phrases):
            writer.writerow({"text": phrase, "source": source, "weight": 1.0, "category": "study_q_train"})


def build() -> dict:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    layout = json.loads(LAYOUT.read_text(encoding="utf-8"))["StudyQ"]
    grid = np.empty((layout["n_rows"], layout["n_cols"]), dtype=object)
    for symbol, (row, col) in layout["grid_map"].items():
        grid[row - 1, col - 1] = symbol.lower()
    with h5py.File(CACHE, "r") as h5:
        run_ids = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in h5["run_id"][:]])
        y = h5["y"][:].astype(np.uint8)
        char_idx = h5["char_idx"][:].astype(np.int32)
        masks = h5["lit_mask"][:].astype(bool)
    cache_runs = set(run_ids)
    if cache_runs != set(registry):
        raise ValueError("Q phrase sources must match the clean classifier training registry exactly")
    by_subject: dict[str, set[str]] = defaultdict(set)
    for run_id in sorted(cache_runs):
        if "_Train" not in run_id or "_Test" in run_id or "_SE003_" in run_id:
            raise ValueError(f"Unsafe Q RAG source run: {run_id}")
        run_rows = np.flatnonzero(run_ids == run_id)
        key_sequence = []
        for char in sorted(set(char_idx[run_rows])):
            rows = run_rows[char_idx[run_rows] == char]
            target_masks = masks[rows[y[rows] == 1]]
            if not len(target_masks):
                raise ValueError(f"No target flashes found for {run_id} character {char}")
            intersection = np.all(target_masks, axis=0)
            if int(intersection.sum()) != 1:
                raise ValueError(f"Target grid key is ambiguous for {run_id} character {char}")
            key_sequence.append(str(grid.reshape(-1)[int(np.flatnonzero(intersection)[0])]))
        phrase = " ".join(key_sequence)
        by_subject[run_id.split("_SE", 1)[0]].add(phrase)
    if len(by_subject) != 36:
        raise ValueError(f"Expected 36 Q subjects, found {len(by_subject)}")
    global_phrases = set().union(*by_subject.values())
    _write_bank(OUT / "phrase_bank_global.csv", global_phrases, "study_q_clean_train")
    counts = {}
    for subject, phrases in sorted(by_subject.items()):
        _write_bank(OUT / "by_subject" / f"phrase_bank_{subject}.csv", phrases, "study_q_clean_train")
        counts[subject] = {"unique_phrases": len(phrases), "phrases_with_whitespace": sum(any(c.isspace() for c in p) for p in phrases)}
    report = {
        "source_registry": str(REGISTRY.relative_to(ROOT)),
        "source_run_count": len(registry),
        "training_cache_sha256": hashlib.sha256(CACHE.read_bytes()).hexdigest(),
        "heldout_session": "SE003",
        "test_labels_read": False,
        "global_unique_phrases": len(global_phrases),
        "unique_phrases_by_subject": counts,
        "phrase_bank_root": str(OUT.relative_to(ROOT)),
        "symbol_encoding": "one grid key label per whitespace-delimited token, derived from clean Train target masks",
        "interpretation": "The bank preserves multi-character grid keys. These are typed key sequences, not conventional natural-language words; benefits require inner validation.",
    }
    (OUT / "build_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(build(), indent=2))
