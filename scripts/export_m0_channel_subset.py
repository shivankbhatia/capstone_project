#!/usr/bin/env python3
"""Fit the locked clean M0 on a channel subset of the D clean cache (exploratory)."""
from pathlib import Path
import argparse, json, sys
import h5py, joblib, numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_classifier_manifest import _fit_final_models  # noqa: E402
from src.models.classifier_interface import EpochFeatureScorer  # noqa: E402

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, default=ROOT / "data/cache/classifier_1_12_no_baseline.h5")
    ap.add_argument("--channels", nargs="+", required=True)
    ap.add_argument("--subset-cache", type=Path, default=ROOT / "data/cache/classifier_1_12_no_baseline_16ch.h5")
    ap.add_argument("--output", type=Path, default=ROOT / "data/processed/m0_16ch_epoch_scorer.pkl")
    a = ap.parse_args()
    if a.subset_cache.exists():
        raise FileExistsError(a.subset_cache)
    with h5py.File(a.cache, "r") as src, h5py.File(a.subset_cache, "w") as dst:
        names = str(src.attrs["channels"]).split(",")
        missing = [c for c in a.channels if c not in names]
        if missing:
            raise ValueError(f"Not in D cache: {missing}")
        idx = np.array([names.index(c) for c in a.channels])
        n, _, t = src["X"].shape
        dX = dst.create_dataset("X", shape=(n, len(idx), t), dtype="f4",
                                chunks=(32, len(idx), t), compression="gzip", compression_opts=1)
        for s in range(0, n, 4096):
            dX[s:s + 4096] = src["X"][s:s + 4096][:, idx, :]
        for key in src.keys():
            if key != "X":
                src.copy(key, dst)
        for k, v in src.attrs.items():
            dst.attrs[k] = v
        dst.attrs["channels"] = ",".join(a.channels)
    lock = json.loads((ROOT / "splits/finalist_lock.json").read_text())
    config = ROOT / "configs/classifier_optimization/preprocessing/baseline_off.json"
    fitted, channels, cfg, sfreq, tmin = _fit_final_models(a.subset_cache, config, lock)
    m0 = fitted["M0"]
    scorer = EpochFeatureScorer(m0["classifier"], m0["scaler"], channels, sfreq, tmin, cfg,
                                temperature=1.0)  # D-32ch temperature does not apply
    joblib.dump(scorer, a.output)
    print(f"Wrote {a.subset_cache} and {a.output}")

if __name__ == "__main__":
    main()