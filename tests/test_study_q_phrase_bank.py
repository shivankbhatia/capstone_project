import csv
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


def _phrases(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["text"] for row in csv.DictReader(handle)}


def test_study_q_phrase_banks_use_only_clean_training_registry():
    root = Path(__file__).resolve().parents[1]
    registry = json.loads((root / "data/processed/study_q_train_registry.json").read_text())
    manifest = json.loads((root / "splits/study_q_manifest.json").read_text())
    excluded = json.loads((root / "splits/study_q_excluded_training_runs.json").read_text())
    heldout = {run for runs in manifest.values() for run in runs}
    excluded_ids = {run for runs in excluded.values() for run in runs}
    assert not (set(registry) & (heldout | excluded_ids))
    layout = json.loads((root / "data/processed/grid_layout.json").read_text())["StudyQ"]
    grid = np.empty((layout["n_rows"], layout["n_cols"]), dtype=object)
    for symbol, (row, col) in layout["grid_map"].items():
        grid[row - 1, col - 1] = symbol.lower()
    with h5py.File(root / "data/cache/study_q_classifier_epochs.h5", "r") as h5:
        run_ids = np.asarray([v.decode() if isinstance(v, bytes) else str(v) for v in h5["run_id"][:]])
        y = h5["y"][:].astype(np.uint8)
        chars = h5["char_idx"][:].astype(np.int32)
        masks = h5["lit_mask"][:].astype(bool)
    assert set(run_ids) == set(registry)
    expected_by_subject = defaultdict(set)
    for run_id in sorted(set(run_ids)):
        rows = np.flatnonzero(run_ids == run_id)
        keys = []
        for char in sorted(set(chars[rows])):
            char_rows = rows[chars[rows] == char]
            positive = masks[char_rows[y[char_rows] == 1]]
            intersection = np.all(positive, axis=0)
            assert intersection.sum() == 1
            keys.append(str(grid.reshape(-1)[np.flatnonzero(intersection)[0]]))
        expected_by_subject[run_id.split("_SE", 1)[0]].add(" ".join(keys))
    expected_global = set().union(*expected_by_subject.values())
    bank_root = root / "data/rag/study_q"
    assert _phrases(bank_root / "phrase_bank_global.csv") == expected_global
    for subject, phrases in expected_by_subject.items():
        assert _phrases(bank_root / "by_subject" / f"phrase_bank_{subject}.csv") == phrases
