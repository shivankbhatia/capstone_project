"""Compute RP + CWT 2D features for already-epoched Study D files.

Reads existing data/processed/StudyD/*-epo.fif (produced by
src/data/batch_preprocess.py -- untouched), computes recurrence plots and
CWT spectrograms per epoch, and writes a companion HDF5 per file:

    data/processed/StudyD/features/<stem>-feat.h5
        recurrence_2d : (N, 8, 2080) float16   -- 64*65/2 = 2080
        cwt_2d        : (N, 8, 30, 64) float16

Space optimization vs. the naive full-montage/native-rate version: RP/CWT
are computed on a canonical 8-channel P300 subset and a time axis decimated
to 64 samples (RP_CWT_N_TIMES in feature_transforms.py). This is
independent of the classifier's native 256 Hz / full-montage data -- the
classifier, decoder, fusion, LLM, and RAG code are untouched.

Epoch order matches the source -epo.fif exactly, so existing metadata
(stimulus_code, char_index, sequence_in_char) still lines up by index.

Usage: python scripts/compute_2d_features.py [--limit N]
"""
import argparse
import gc
from pathlib import Path

import h5py
import mne

from src.preprocessing.feature_transforms import (
    RP_CWT_N_TIMES,
    _decimate_time,
    compute_cwt_spectrograms,
    compute_recurrence_plots,
    select_rp_cwt_channels,
)

STUDYD_DIR = Path("data/processed/StudyD")
OUT_DIR = STUDYD_DIR / "features"


def process_file(fif_path: Path, batch_size: int = 100):
    epochs = mne.read_epochs(str(fif_path), preload=True, verbose=False)
    data = epochs.get_data(copy=False)  # (N, C, T), native sfreq, full montage
    sfreq = epochs.info["sfreq"]

    # Space optimization: subset to the canonical 8 P300 channels, then
    # decimate the time axis to RP_CWT_N_TIMES. Native data / sfreq used
    # by the classifier elsewhere is untouched -- this only affects what
    # gets computed here.
    ch_idx = select_rp_cwt_channels(epochs.ch_names)
    data_sub = data[:, ch_idx, :]

    n_times_in = data_sub.shape[-1]
    data_dec = _decimate_time(data_sub, RP_CWT_N_TIMES)
    effective_sfreq = sfreq * (data_dec.shape[-1] / n_times_in)

    rp = compute_recurrence_plots(data_dec, batch_size=batch_size)
    cwt = compute_cwt_spectrograms(data_dec, sfreq=effective_sfreq, batch_size=batch_size)

    labels = epochs.events[:, 2].astype("int32")  # 0=non-target, 1=target

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{fif_path.stem}-feat.h5"
    with h5py.File(out_path, "w") as h5f:
        h5f.create_dataset("recurrence_2d", data=rp, compression="lzf")
        h5f.create_dataset("cwt_2d", data=cwt, compression="lzf")
        h5f.create_dataset("labels", data=labels)
        h5f.attrs["session_id"] = fif_path.stem.replace("-epo", "")
        h5f.attrs["n_epochs"] = data_dec.shape[0]
        h5f.attrs["n_channels"] = data_dec.shape[1]
        h5f.attrs["channel_names"] = [epochs.ch_names[i] for i in ch_idx]
        h5f.attrs["n_times"] = data_dec.shape[2]
        h5f.attrs["native_sfreq"] = sfreq
        h5f.attrs["effective_sfreq"] = effective_sfreq

    del epochs, data, data_sub, data_dec, rp, cwt
    gc.collect()
    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                         help="Process only the first N files (for a quick check).")
    parser.add_argument("--batch_size", type=int, default=100)
    args = parser.parse_args()

    fif_files = sorted(STUDYD_DIR.glob("*-epo.fif"))
    if args.limit:
        fif_files = fif_files[:args.limit]

    print(f"Found {len(fif_files)} epoch files under {STUDYD_DIR}")
    for i, fp in enumerate(fif_files, 1):
        print(f"[{i}/{len(fif_files)}] {fp.name}")
        try:
            out = process_file(fp, batch_size=args.batch_size)
            print(f"  -> {out}")
        except Exception as e:
            print(f"  ERROR: {e}")