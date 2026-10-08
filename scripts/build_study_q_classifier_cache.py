#!/usr/bin/env python3
"""Build Study Q training epochs only, with group-flash membership masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.batch_preprocess import parse_bigp3bci_edf  # noqa: E402
from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint,
    load_heldout_run_ids,
)

DATA = ROOT / "data/raw/bigP3BCI_dataset/bigP3BCI-data-backup/StudyQ"
MANIFEST = ROOT / "splits/study_q_manifest.json"
EXCLUDED = ROOT / "splits/study_q_excluded_training_runs.json"


def build(output: Path, *, max_runs: int | None = None) -> dict:
    heldout = load_heldout_run_ids(MANIFEST)
    excluded_by_subject = json.loads(EXCLUDED.read_text(encoding="utf-8"))
    excluded_training = {run for runs in excluded_by_subject.values() for run in runs}
    all_train = sorted(p for p in DATA.rglob("*.edf")
                       if "/Train/" in p.as_posix() and not p.name.startswith("._"))
    same_session_train = {p.stem for p in all_train if "_SE003_" in p.stem}
    if same_session_train != excluded_training:
        raise ValueError("Study Q SE003 Train exclusion file does not match raw path inventory")
    selected = [p for p in all_train if p.stem not in excluded_training]
    assert_training_pool_disjoint(selected, heldout)
    if any("_Test" in p.stem or "_SE003_" in p.stem for p in selected):
        raise ValueError("Refusing to include Test or held-out-session data in Q training cache")
    if max_runs is not None:
        selected = selected[:max_runs]
    if not selected:
        raise RuntimeError("No Study Q training runs remain after session exclusion")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Q cache: {output}")

    total = 0
    channels = None
    with h5py.File(output, "w") as h5:
        datasets = {}
        for index, path in enumerate(selected, start=1):
            epochs, _, info = parse_bigp3bci_edf(
                path, tmin=-0.1, tmax=0.8, l_freq=1.0, h_freq=12.0,
                baseline_correction=False, decim=2, verbose=False,
            )
            metadata = epochs.metadata
            required = {"lit_mask", "char_index", "sequence_in_char"}
            if metadata is None or not required.issubset(metadata.columns):
                raise ValueError(f"{path}: missing group-flash or sequence metadata")
            X = epochs.get_data(copy=False).astype(np.float32, copy=False)
            y = epochs.events[:, 2].astype(np.uint8, copy=False)
            masks = np.asarray([[int(bit) for bit in mask]
                                for mask in metadata["lit_mask"].astype(str)], dtype=np.uint8)
            if masks.shape != (len(y), 72):
                raise ValueError(f"{path}: unexpected Q mask shape {masks.shape}")
            seq = metadata["sequence_in_char"].to_numpy(dtype=np.int32)
            char = metadata["char_index"].to_numpy(dtype=np.int32)
            if np.any(seq < 0) or np.any(char < 0):
                raise ValueError(f"{path}: incomplete selected-target segmentation")
            if not info["group_flash"] or int(info["flashes_per_seq"]) != 12:
                raise ValueError(f"{path}: expected 12-flash group paradigm")

            if channels is None:
                channels = list(epochs.ch_names)
                n_channels, n_times = X.shape[1:]
                datasets["X"] = h5.create_dataset(
                    "X", shape=(0, n_channels, n_times), maxshape=(None, n_channels, n_times),
                    dtype="f4", chunks=(32, n_channels, n_times), compression="gzip", compression_opts=1,
                )
                for name, dtype, tail in (
                    ("y", "u1", ()), ("lit_mask", "u1", (72,)),
                    ("seq_idx", "i4", ()), ("char_idx", "i4", ()),
                ):
                    datasets[name] = h5.create_dataset(
                        name, shape=(0,) + tail, maxshape=(None,) + tail, dtype=dtype,
                        chunks=(max(32, min(8192, len(y))),) + tail, compression="gzip",
                    )
                for name in ("subject", "session", "run_id", "condition"):
                    datasets[name] = h5.create_dataset(
                        name, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"),
                        chunks=(max(32, min(8192, len(y))),), compression="gzip",
                    )
            if X.shape[1:] != datasets["X"].shape[1:]:
                raise ValueError(f"{path}: inconsistent channel/time shape {X.shape}")
            subject = path.parts[-5]
            session = path.parts[-4]
            condition = path.parts[-2]
            values = {
                "X": X, "y": y, "lit_mask": masks, "seq_idx": seq, "char_idx": char,
                "subject": np.full(len(y), subject, dtype=object),
                "session": np.full(len(y), session, dtype=object),
                "run_id": np.full(len(y), path.stem, dtype=object),
                "condition": np.full(len(y), condition, dtype=object),
            }
            start, stop = total, total + len(y)
            for name, value in values.items():
                ds = datasets[name]
                ds.resize((stop,) + ds.shape[1:])
                ds[start:stop] = value
            total = stop
            if index % 20 == 0 or index == len(selected):
                print(f"Cached {index}/{len(selected)} clean Q Train runs ({total} epochs)", flush=True)
            del epochs
        h5.attrs["manifest_sha256"] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
        h5.attrs["excluded_train_sha256"] = hashlib.sha256(EXCLUDED.read_bytes()).hexdigest()
        h5.attrs["training_runs"] = len(selected)
        h5.attrs["heldout_session"] = "SE003"
        h5.attrs["channels"] = ",".join(channels)
        h5.attrs["sfreq"] = 128.0
        h5.attrs["tmin"] = -0.1
        h5.attrs["l_freq"] = 1.0
        h5.attrs["h_freq"] = 12.0
        h5.attrs["decim"] = 2
        h5.attrs["baseline_correction"] = False
        h5.attrs["flashes_per_sequence"] = 12
    return {"cache": str(output), "runs": len(selected), "epochs": total,
            "manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
            "heldout_test_data_read": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data/cache/study_q_classifier_epochs.h5")
    parser.add_argument("--max-runs", type=int, help="Small train-only smoke run; never includes SE003")
    args = parser.parse_args()
    print(json.dumps(build(args.output, max_runs=args.max_runs), indent=2))
