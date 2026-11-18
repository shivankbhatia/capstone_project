import numpy as np
import pytest

from prototype.server import ReplayService


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


@pytest.mark.parametrize("method,args", [
    ("replay", ("sample", "compact")),
    ("simulate", ("A", "rag")),
])
def test_removed_arms_are_rejected(method, args):
    with pytest.raises(ValueError, match="arm must be"):
        getattr(ReplayService(), method)(*args)
