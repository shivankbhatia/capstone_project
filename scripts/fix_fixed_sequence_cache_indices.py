#!/usr/bin/env python3
"""Correct fixed-RC cache grouping: 10 row/column cycles per character."""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def fix_cache(path: Path, apply: bool = False) -> dict:
    mode = "r+" if apply else "r"
    report = {"path": str(path), "runs": 0, "epochs": 0,
              "sequences_per_character": 10, "characters_by_run": {}}
    with h5py.File(path, mode) as h5:
        runs = h5["run_id"][:].astype(str)
        seq = h5["seq_idx"][:].astype(np.int32)
        char = h5["char_idx"][:].astype(np.int32)
        for run_id in sorted(set(runs)):
            idx = np.flatnonzero(runs == run_id)
            n_seq, rem = divmod(len(idx), 17)
            if rem:
                raise ValueError(f"{run_id}: {len(idx)} epochs is not whole 17-flash sequences")
            if n_seq % 10:
                raise ValueError(f"{run_id}: {n_seq} sequences is not divisible into 10-sequence chars")
            seq[idx] = (np.arange(len(idx), dtype=np.int32) // 17) % 10
            char[idx] = np.arange(len(idx), dtype=np.int32) // (17 * 10)
            report["characters_by_run"][run_id] = n_seq // 10
        report["runs"] = len(report["characters_by_run"])
        report["epochs"] = len(runs)
        if apply:
            h5["seq_idx"][:] = seq
            h5["char_idx"][:] = char
            h5.attrs["fixed_sequences_per_character"] = 10
            h5.attrs["fixed_sequence_grouping"] = "10 sequences; 17 row/column flashes per sequence"
    return report


def main():
    p = argparse.ArgumentParser()
    p.add_argument("caches", nargs="+", type=Path)
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()
    for path in args.caches:
        print(fix_cache(path, apply=args.apply))


if __name__ == "__main__":
    main()
