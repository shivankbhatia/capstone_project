#!/usr/bin/env python3
"""Add per-subject AUCs to nested OOF calibration summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.screen_shrinkage_lda import _decode_strings, _load_metadata


def run(scores_path: Path, cache_path: Path, summary_path: Path):
    with h5py.File(cache_path, "r") as h5:
        meta = _load_metadata(h5)
    with np.load(scores_path, allow_pickle=False) as stored:
        arrays = {name: stored[name] for name in stored.files}
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    subjects = _decode_strings(meta["subject"])
    y = meta["y"]
    for name in list(summary):
        if name + "_raw" not in arrays or name + "_calibrated" not in arrays:
            continue
        for suffix, key in (("raw", "per_subject_raw_epoch_auc"),
                            ("calibrated", "per_subject_calibrated_epoch_auc")):
            logits = arrays[name + "_" + suffix]
            valid_all = np.isfinite(logits)
            values = {}
            for subject in sorted(set(subjects)):
                mask = valid_all & (subjects == subject)
                if len(np.unique(y[mask])) == 2:
                    values[subject] = float(roc_auc_score(y[mask], logits[mask]))
            summary[name][key] = values
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    run(args.scores, args.cache, args.summary)


if __name__ == "__main__":
    main()
