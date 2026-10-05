#!/usr/bin/env python3
"""Reproduce the compact row-focus arm on adaptive-metadata sessions only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from prototype.server import ROOT, ReplayService


# Held-out samples with ``adaptive_index_map`` available. The four omitted
# legacy fixed-block sessions are D_01_RC_Train06, D_03_RC_Train06,
# D_04_RC_Train04, and D_04_RC_Train06.
ADAPTIVE_SESSION_IDS = (
    "D_01_SE001_DynBigram_Test03",
    "D_01_SE001_DynBigram_Test04",
    "D_01_SE001_Dyn_Test02",
    "D_02_SE001_DynBigram_Test05",
    "D_02_SE001_Dyn_Test01",
    "D_02_SE001_Dyn_Test06",
    "D_03_SE001_DynBigram_Test03",
    "D_03_SE001_DynBigram_Test08",
    "D_03_SE001_Dyn_Test01",
    "D_03_SE001_Dyn_Test04",
    "D_04_SE001_DynBigram_Test03",
    "D_04_SE001_DynBigram_Test05",
    "D_04_SE001_Dyn_Test07",
    "D_05_SE001_Dyn_Test04",
)


def main() -> None:
    service = ReplayService()
    targets = {sample["id"]: sample["target"] for sample in service.samples()}
    classifier = service._load_classifier()
    runs = []
    for session_id in ADAPTIVE_SESSION_IDS:
        result = service._decode_row_focus(
            targets[session_id],
            str(ROOT / "data/processed/StudyD" / f"{session_id}-epo.fif"),
            classifier,
            session_id,
        )
        assert result["source"] == "study_d_row_focus_replay"
        runs.append(result)

    characters = sum(run["summary"]["characters"] for run in runs)
    flashes = sum(run["summary"]["sequences"] for run in runs)
    correct = sum(run["summary"]["correct"] for run in runs)
    aggregate = {
        "sessions": len(runs),
        "characters": characters,
        "correct": correct,
        "total_flashes": flashes,
        "mean_characters_per_session": characters / len(runs),
        "flashes_per_session": flashes / len(runs),
        "flashes_per_character": flashes / characters,
        "accuracy": 100 * correct / characters,
    }
    output = ROOT / "dead_arms" / "results" / "row_focus_adaptive_subset.json"
    output.write_text(json.dumps({"session_ids": ADAPTIVE_SESSION_IDS, "runs": runs,
                                  "aggregate": aggregate}, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
