import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from scripts.build_phrase_bank_from_registry import is_test, subject_id_from_session_id


ROOT = Path(__file__).resolve().parents[1]


def _bank_words(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["text"] for row in csv.DictReader(handle)}


def test_phrase_banks_are_deterministic_training_pool_only():
    registry = json.loads((ROOT / "data/processed/ground_truth_registry.json").read_text())
    manifest = json.loads((ROOT / "splits/heldout_manifest.json").read_text())
    heldout = {run for runs in manifest.values() for run in runs}
    assert not (set(registry) & heldout)

    expected_global = set()
    expected_by_subject = defaultdict(set)
    for run_id, target in registry.items():
        if not isinstance(target, str) or not target.strip() or is_test(run_id):
            continue
        word = target.strip().upper()
        expected_global.add(word)
        expected_by_subject[subject_id_from_session_id(run_id)].add(word)

    assert _bank_words(ROOT / "data/rag/phrase_bank_global.csv") == expected_global
    for subject, words in expected_by_subject.items():
        assert _bank_words(ROOT / f"data/rag/by_subject/phrase_bank_{subject}.csv") == words


def test_special_key_map_covers_all_multichar_study_d_keys():
    layout = json.loads((ROOT / "data/processed/grid_layout.json").read_text())["StudyD"]
    key_map = json.loads((ROOT / "configs/llm/special_key_map.json").read_text())
    multichar = {key for key in layout["grid_map"] if len(key) > 1}
    assert multichar <= set(key_map)
