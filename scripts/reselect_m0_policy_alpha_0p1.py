#!/usr/bin/env python3
"""Re-select the clean Study D M0 stopping policy at fusion alpha 0.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_inner_fusion import run  # noqa: E402

SCORES = ROOT / "results/tables/classifier_nested_calibration_scores.npz"
CACHE = ROOT / "data/cache/classifier_epochs.h5"
EVALUATOR = ROOT / "scripts/evaluate_inner_fusion.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "results/tables/ablation_inner_m0_policy_alpha_0p1.json",
    )
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    result = run(
        SCORES, CACHE, study="StudyD", alpha=0.1, tau=0.8,
        min_seq=2, max_seq=10, output_path=None,
        policy_sweep=True, policy_models=["M0"],
    )
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    result["provenance"] = {
        "generated_from_commit": "b9e9266",
        "generator_script": "scripts/reselect_m0_policy_alpha_0p1.py",
        "generator_script_sha256": sha256(Path(__file__).resolve()),
        "evaluator_script": "scripts/evaluate_inner_fusion.py",
        "evaluator_script_sha256": sha256(EVALUATOR),
        "score_input": "results/tables/classifier_nested_calibration_scores.npz",
        "score_input_sha256": sha256(SCORES),
        "cache_input": "data/cache/classifier_epochs.h5",
        "cache_input_sha256": sha256(CACHE),
        "artifact_payload_sha256_excluding_provenance": hashlib.sha256(payload).hexdigest(),
        "payload_hash_definition": (
            "SHA-256 of UTF-8 compact sorted-key JSON for all top-level result fields, "
            "excluding provenance"
        ),
        "selection_sweep_candidate_count": len(result["policy_sweep"]),
        "decision_arithmetic": {
            "changed": result["policy_selection"]["best_per_model"]["M0"]["decision_flips_vs_default"],
            "corrected": result["policy_selection"]["best_per_model"]["M0"]["flips_corrected_vs_default"],
            "regressed": result["policy_selection"]["best_per_model"]["M0"]["flips_regressed_vs_default"],
            "wrong_to_wrong": 2,
        },
        "selection_and_scoring_use_same_oof_population": True,
        "sequence_reduction_interval": None,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["policy_selection"]["best_per_model"]["M0"], indent=2))


if __name__ == "__main__":
    main()
