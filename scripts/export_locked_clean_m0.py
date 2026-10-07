#!/usr/bin/env python3
"""Fit and export the locked clean M0 model for the GUI/runtime score API."""
from pathlib import Path
import json
import sys

import h5py
import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_classifier_manifest import _fit_final_models  # noqa: E402
from src.models.classifier_interface import EpochFeatureScorer  # noqa: E402


def main():
    lock_path = ROOT / "splits/finalist_lock.json"
    lock = json.loads(lock_path.read_text())
    cache = ROOT / "data/cache/classifier_1_12_no_baseline.h5"
    config = ROOT / "configs/classifier_optimization/preprocessing/baseline_off.json"
    fitted, channels, cfg, sfreq, tmin = _fit_final_models(cache, config, lock)
    m0 = fitted["M0"]
    scorer = EpochFeatureScorer(
        m0["classifier"], m0["scaler"], channels, sfreq, tmin, cfg,
        temperature=m0["temperature"],
    )
    output = ROOT / "data/processed/clean_m0_epoch_scorer.pkl"
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scorer, output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
