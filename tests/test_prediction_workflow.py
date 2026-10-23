from unittest.mock import patch

import pytest

from hematology.constants import FEATURES
from hematology.data import DataValidationError
from hematology.history import PredictionHistory
from hematology.service import build_system, predict_and_save


@pytest.fixture(scope="module")
def system():
    return build_system()


@pytest.mark.parametrize("bad_value", [0, -1, 1e12, "sadaa", "", None, float("nan"), float("inf")])
def test_invalid_input_never_predicts_or_saves(system, tmp_path, bad_value):
    history = PredictionHistory(tmp_path / "history.sqlite3")
    record = system.examples["Healthy"][0].copy()
    record["HGB"] = bad_value
    with patch.object(system.model, "explain_one") as predict:
        with pytest.raises(DataValidationError):
            predict_and_save(system, history, record, units_confirmed=True)
        predict.assert_not_called()
    assert history.list().empty


def test_units_gate_blocks_inference_and_history(system, tmp_path):
    history = PredictionHistory(tmp_path / "history.sqlite3")
    with patch.object(system.model, "explain_one") as predict:
        with pytest.raises(DataValidationError, match="Confirm"):
            predict_and_save(system, history, system.examples["Healthy"][0], units_confirmed=False)
        predict.assert_not_called()
    assert history.list().empty


def test_valid_prediction_saves_all_features_and_mapped_risk(system, tmp_path):
    history = PredictionHistory(tmp_path / "history.sqlite3")
    saved = predict_and_save(system, history, system.examples["Healthy"][0], units_confirmed=True)
    row = history.list().iloc[0]
    assert row["risk_class"] == "Class 1 — Low Risk"
    assert row["id"] == saved.history_id
    assert all(row[name] == saved.record[name] for name in FEATURES)
