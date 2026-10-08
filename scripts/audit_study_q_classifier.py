#!/usr/bin/env python3
"""Inventory existing Study Q classifier artifacts without opening Q EEG."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/tables/study_q_classifier_parity.json"
SKIP = {".git", ".venv", "__pycache__"}
NEEDLES = ("studyq", "study_q", "classifier_q", "swlda_model_q", "q_classifier")


def audit() -> dict:
    candidates = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or any(part in SKIP for part in path.parts):
            continue
        try:
            rel = path.relative_to(ROOT)
        except ValueError:
            continue
        if "data/raw" in rel.as_posix():
            continue
        name = path.name.lower()
        if any(needle in name for needle in NEEDLES):
            candidates.append(str(rel))
    model_artifacts = [
        name for name in candidates
        if Path(name).suffix.lower() in {".pkl", ".joblib", ".pt", ".pth", ".onnx"}
    ]
    report = {
        "study": "StudyQ",
        "audit_scope": "repository-visible filenames only; no model or EEG files loaded",
        "candidate_paths": sorted(candidates),
        "classifier_artifacts_found": sorted(model_artifacts),
        "parity_status": "unavailable_no_persisted_q_classifier_artifact" if not model_artifacts else "artifact_requires_provenance_audit",
        "training_run_manifest": None,
        "preprocessing_provenance": None,
        "calibration_scope": None,
        "q_specific_tuning_provenance": None,
        "interpretation": "Any earlier Q classifier numbers remain exploratory; training overlap, preprocessing, calibration, and tuning cannot be certified from the available artifact inventory.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
