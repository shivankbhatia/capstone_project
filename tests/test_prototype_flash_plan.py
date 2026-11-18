import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from prototype.server import ReplayService
from run_pipeline import real_eeg_classifier_flash_stream


def test_heldout_labels_require_explicit_service_opt_in(monkeypatch, tmp_path):
    registry = tmp_path / "registry.json"
    split = tmp_path / "split.json"
    vault = tmp_path / "vault.json"
    registry.write_text("{}", encoding="utf-8")
    split.write_text('["heldout-session"]', encoding="utf-8")
    monkeypatch.setattr("prototype.server.REGISTRY_PATH", registry)
    monkeypatch.setattr("prototype.server.TEST_SPLIT_PATH", split)
    monkeypatch.setattr("prototype.server.GROUND_TRUTH_VAULT_PATH", vault)

    assert ReplayService().samples() == []
    with pytest.raises(RuntimeError, match="label vault is unavailable"):
        ReplayService(allow_heldout_eval=True).samples()


def test_flash_plan_preserves_recorded_order_and_exposes_classifier_score():
    service = ReplayService()
    prior = np.full(len(service.char_list), 1 / len(service.char_list))

    plan = service._flash_plan(
        [
            {"stimulus_code": 10, "target_probability": 0.12},
            {"stimulus_code": 1, "target_probability": 0.81},
            {"stimulus_code": 11, "target_probability": 0.09},
        ],
        prior,
        sequence=1,
    )

    assert [flash["label"] for flash in plan] == ["COLUMN 1", "ROW 1", "COLUMN 2"]
    assert plan[1]["keys"] == list("ABCDEFGH")
    assert plan[1]["classifier_target"] is True


def test_later_sequence_does_not_order_groups_by_language_prior_mass():
    service = ReplayService()
    prior = np.zeros(len(service.char_list))
    prior[service.char_list.index("F")] = 1.0  # F is in row 1 and column 6.

    plan = service._flash_plan(
        [
            {"stimulus_code": 11, "target_probability": 0.2},  # column 2
            {"stimulus_code": 1, "target_probability": 0.2},   # row 1
            {"stimulus_code": 15, "target_probability": 0.2},  # column 6
        ],
        prior,
        sequence=2,
    )

    assert [flash["label"] for flash in plan] == ["COLUMN 2", "ROW 1", "COLUMN 6"]


def test_service_uses_fallback_grid_without_processed_data(monkeypatch, tmp_path):
    monkeypatch.setattr("prototype.server.GRID_PATH", tmp_path / "missing-grid.json")
    monkeypatch.setattr("prototype.server.REGISTRY_PATH", tmp_path / "missing-registry.json")
    monkeypatch.setattr("prototype.server.TEST_SPLIT_PATH", tmp_path / "missing-split.json")

    service = ReplayService()

    assert service.using_fallback_grid is True
    assert service.samples() == []
    assert service.simulate("A", "eeg_only")["source"] == "custom_simulation"


def test_fallback_grid_keys_are_lm_mapped_or_alphanumeric():
    with open("configs/studyd_grid_layout.json", encoding="utf-8") as file:
        layout = json.load(file)
    with open("configs/llm/special_key_map.json", encoding="utf-8") as file:
        special = json.load(file)

    keys = layout["StudyD"]["grid_map"]
    unmapped = {key for key in keys if key not in special and not key.isalnum()}

    assert unmapped == {"!", "'", "+", ",", "-", "/", ";", "=", "?", "\\"}


def test_locked_special_key_map_hash_is_unchanged():
    path = Path("configs/llm/special_key_map.json")

    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        "eb1eadd6e21ab4c290611a97284a4745e9bd329ec1a9c85c2f3f70d19397088e"
    )
    assert ReplayService().health()["locked_special_key_map"]["matches_lock"] is True


def test_fallback_grid_matches_processed_grid_when_available():
    fallback_path = Path("configs/studyd_grid_layout.json")
    processed_path = Path("data/processed/grid_layout.json")
    if not processed_path.exists():
        pytest.skip("processed Study D grid is unavailable in this checkout")

    fallback = json.loads(fallback_path.read_text(encoding="utf-8"))["StudyD"]
    processed = json.loads(processed_path.read_text(encoding="utf-8"))["StudyD"]

    assert fallback["grid_map"] == processed["grid_map"]


def test_real_flash_stream_preserves_epoch_stimulus_order(monkeypatch):
    import run_pipeline

    class Epochs:
        def __init__(self, codes):
            self.metadata = pd.DataFrame({
                "stimulus_code": codes,
                "char_index": [0] * len(codes),
                "sequence_in_char": [1] * len(codes),
            })

        def __len__(self):
            return len(self.metadata)

        def __getitem__(self, rows):
            return Epochs(self.metadata.iloc[list(rows)]["stimulus_code"].tolist())

        def pick(self, *_args, **_kwargs):
            return self

        def get_data(self, copy=False):
            return np.zeros((len(self.metadata), 1, 2))

    class Classifier:
        def calibrated_logits(self, values):
            return np.arange(len(values), dtype=float)

    path = "recorded-order-test.fif"
    monkeypatch.setattr(run_pipeline, "_epoch_cache", {path: Epochs([10, 1, 11])})
    monkeypatch.setattr(run_pipeline, "_adaptive_index_cache", {path: {(0, 1): np.array([0, 1, 2])}})

    flashes = real_eeg_classifier_flash_stream(path, 0, 1, Classifier(), 9, 8, 17)

    assert [flash["stimulus_code"] for flash in flashes] == [10, 1, 11]


@pytest.mark.parametrize("method,args", [
    ("replay", ("sample", "compact")),
    ("simulate", ("A", "rag")),
])
def test_removed_arms_are_rejected(method, args):
    with pytest.raises(ValueError, match="arm must be"):
        getattr(ReplayService(), method)(*args)
