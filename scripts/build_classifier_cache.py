#!/usr/bin/env python3
"""Build a chunked HDF5 epoch cache from the clean, non-held-out pool only."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import sys
from typing import Optional

import h5py
import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.classifier_protocol import (  # noqa: E402
    assert_training_pool_disjoint,
    load_heldout_run_ids,
    run_id_from_epoch_path,
)


def _metadata_array(epochs, column: str, default: int) -> np.ndarray:
    if epochs.metadata is not None and column in epochs.metadata.columns:
        return epochs.metadata[column].to_numpy()
    return np.full(len(epochs), default)


def _condition(run_id: str) -> str:
    if "DynBigram" in run_id:
        return "DynBigram"
    if "Dyn" in run_id:
        return "Dyn"
    if "RC_Train" in run_id:
        return "RC/Train"
    return "unknown"


def _sequence_and_character_indices(epochs, run_id: str) -> tuple[np.ndarray, np.ndarray]:
    md = epochs.metadata
    if md is not None and {"char_index", "sequence_in_char"}.issubset(md.columns):
        char_idx = md["char_index"].to_numpy(dtype=np.int32)
        seq_idx = md["sequence_in_char"].to_numpy(dtype=np.int32)
        return seq_idx, char_idx

    # Fixed RC calibration files contain 17 row/column flashes per sequence
    # and 10 sequences per character (20 target flashes: row + column for
    # each sequence). The cache records 0-based sequence indices.
    n_rows, n_cols = 9, 8
    flashes_per_sequence = n_rows + n_cols
    event_order = np.arange(len(epochs), dtype=np.int32)
    sequences_per_char = 10
    seq_idx = (event_order // flashes_per_sequence) % sequences_per_char
    char_idx = event_order // (flashes_per_sequence * sequences_per_char)
    return seq_idx, char_idx


def build_cache(
    epoch_dir: Path, manifest_path: Path, output: Path, *, source: str = "epochs",
    raw_dir: Optional[Path] = None, baseline_correction: bool = True,
    l_freq: float = 0.1, h_freq: float = 30.0,
) -> None:
    heldout = load_heldout_run_ids(manifest_path)
    all_paths = sorted(
        path for path in epoch_dir.glob("*-epo.fif")
        if not path.name.startswith("._")
    )
    training_paths = [
        path for path in all_paths
        if run_id_from_epoch_path(path) not in heldout and "_Train" in path.name
    ]
    assert_training_pool_disjoint(training_paths, heldout)
    if not training_paths:
        raise RuntimeError("No clean training epochs found")

    raw_paths = {}
    if source == "raw":
        if raw_dir is None:
            raise ValueError("--raw-dir is required when --source raw")
        for raw_path in raw_dir.rglob("*.edf"):
            if raw_path.name.startswith("._"):
                continue
            raw_paths.setdefault(raw_path.stem, raw_path)

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing cache: {output}")

    n_channels = n_times = None
    channel_names = None
    sfreq = None
    tmin = tmax = None
    total = 0
    with h5py.File(output, "w") as h5:
        datasets = {}
        for path in training_paths:
            run_id = run_id_from_epoch_path(path)
            if source == "raw":
                from src.data.batch_preprocess import parse_bigp3bci_edf
                raw_path = raw_paths.get(run_id)
                if raw_path is None:
                    raise FileNotFoundError(f"No raw EDF found for {run_id} under {raw_dir}")
                epochs, _, _ = parse_bigp3bci_edf(
                    raw_path, tmin=-0.1, tmax=0.8, l_freq=l_freq, h_freq=h_freq,
                    baseline_correction=baseline_correction, verbose=False,
                )
            else:
                epochs = mne.read_epochs(path, preload=True, verbose="ERROR")
            X = epochs.get_data(copy=False).astype(np.float32, copy=False)
            y = epochs.events[:, 2].astype(np.uint8, copy=False)
            if n_channels is None:
                n_channels, n_times = X.shape[1:]
                channel_names = list(epochs.ch_names)
                sfreq = float(epochs.info["sfreq"])
                tmin, tmax = float(epochs.tmin), float(epochs.tmax)
                shape = (0, n_channels, n_times)
                maxshape = (None, n_channels, n_times)
                datasets["X"] = h5.create_dataset(
                    "X", shape=shape, maxshape=maxshape, dtype="f4",
                    chunks=(min(32, len(X)), n_channels, n_times), compression="gzip",
                    compression_opts=1,
                )
                for name, dtype in (("y", "u1"), ("seq_idx", "i4"),
                                    ("stim_code", "i4"), ("char_idx", "i4")):
                    datasets[name] = h5.create_dataset(
                        name, shape=(0,), maxshape=(None,), dtype=dtype,
                        chunks=(max(256, min(8192, len(X))),), compression="gzip",
                    )
                for name in ("subject", "run_id", "condition"):
                    datasets[name] = h5.create_dataset(
                        name, shape=(0,), maxshape=(None,), dtype=h5py.string_dtype("utf-8"),
                        chunks=(max(256, min(8192, len(X))),), compression="gzip",
                    )

            if X.shape[1:] != (n_channels, n_times):
                raise ValueError(f"Inconsistent epoch shape in {path}: {X.shape}")
            seq_idx, char_idx = _sequence_and_character_indices(epochs, run_id)
            stim_code = _metadata_array(epochs, "stimulus_code", -1).astype(np.int32)
            subject = run_id.split("_SE", 1)[0]
            values = {
                "X": X,
                "y": y,
                "subject": np.full(len(y), subject, dtype=object),
                "run_id": np.full(len(y), run_id, dtype=object),
                "condition": np.full(len(y), _condition(run_id), dtype=object),
                "seq_idx": seq_idx,
                "stim_code": stim_code,
                "char_idx": char_idx,
            }
            start, stop = total, total + len(y)
            for name, value in values.items():
                ds = datasets[name]
                ds.resize((stop,) + ds.shape[1:])
                ds[start:stop] = value
            total = stop
            print(f"Cached {run_id}: {len(y)} epochs (total {total})")
            del epochs

        h5.attrs["manifest_path"] = str(manifest_path)
        h5.attrs["manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        h5.attrs["heldout_run_count"] = len(heldout)
        h5.attrs["training_run_count"] = len(training_paths)
        h5.attrs["channels"] = ",".join(channel_names)
        h5.attrs["sfreq"] = sfreq
        h5.attrs["tmin"] = tmin
        h5.attrs["tmax"] = tmax
        h5.attrs["source_filter"] = f"{l_freq if source == 'raw' else 0.1}-{h_freq if source == 'raw' else 30.0} Hz"
        h5.attrs["l_freq"] = float(l_freq if source == "raw" else 0.1)
        h5.attrs["h_freq"] = float(h_freq if source == "raw" else 30.0)
        h5.attrs["baseline_correction"] = bool(baseline_correction if source == "raw" else True)
    print(f"Wrote {total} epochs to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=Path, default=ROOT / "data/processed/StudyD")
    parser.add_argument("--manifest", type=Path, default=ROOT / "splits/heldout_manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/cache/classifier_epochs.h5")
    parser.add_argument("--source", choices=("epochs", "raw"), default="epochs")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/raw/bigP3BCI_dataset")
    parser.add_argument("--no-baseline-correction", action="store_true")
    parser.add_argument("--l-freq", type=float, default=0.1)
    parser.add_argument("--h-freq", type=float, default=30.0)
    args = parser.parse_args()
    build_cache(
        args.epochs, args.manifest, args.output, source=args.source,
        raw_dir=args.raw_dir if args.source == "raw" else None,
        baseline_correction=not args.no_baseline_correction,
        l_freq=args.l_freq, h_freq=args.h_freq,
    )


if __name__ == "__main__":
    main()
