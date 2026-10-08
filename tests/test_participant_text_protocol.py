import json

import pytest

from scripts.prepare_participant_text_sample import prepare, read_sample, split_text


def test_participant_text_splits_chronologically_and_bank_excludes_eval_tail(tmp_path):
    text = " ".join(f"word{i}" for i in range(750)) + " contact alice@example.org +1 (555) 123-4567"
    source = tmp_path / "sample.txt"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "participant_local" / "P001"

    counts = prepare(source, output)
    payload = json.loads((output / "splits.json").read_text(encoding="utf-8"))
    bank = (output / "phrase_bank_global.csv").read_text(encoding="utf-8")

    assert counts["bank_training"] >= 420
    assert counts["gate_tuning"] >= 140
    assert counts["evaluation"] >= 140
    assert "alice@example.org" not in json.dumps(payload)
    assert "+1 (555) 123-4567" not in json.dumps(payload)
    assert "[PHONE]" in json.dumps(payload)
    assert "word749" in payload["evaluation_text"]
    assert "word749" not in bank


def test_participant_text_requires_minimum_and_sorts_timestamped_jsonl(tmp_path):
    with pytest.raises(ValueError):
        split_text("too short", min_tokens=700)
    records = [
        {"timestamp": "2026-02-02T00:00:00Z", "text": "later"},
        {"timestamp": "2026-02-01T00:00:00Z", "text": "earlier"},
    ]
    path = tmp_path / "ordered.jsonl"
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    assert read_sample(path) == "earlier\nlater"
