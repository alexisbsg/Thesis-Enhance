"""Application-facing orchestration for training and prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .constants import DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_PATH, FEATURES, TARGET_COLUMN
from .data import (
    DataAudit,
    DataValidationError,
    audit_dataset,
    historical_split,
    load_dataset,
    validate_record,
    validate_training_data,
)
from .examples import build_example_records
from .history import PredictionHistory
from .interpretation import risk_for_label
from .models import NumericalID3Classifier, PredictionExplanation


@dataclass(frozen=True)
class SavedPrediction:
    explanation: PredictionExplanation
    record: dict[str, float]
    history_id: int


def predict_and_save(
    system: SystemBundle,
    history: PredictionHistory,
    record: dict[str, object],
    *,
    units_confirmed: bool,
    input_source: str = "manual",
) -> SavedPrediction:
    """Validate all 14 fields before model inference or any history write.

    Keep this gate in the service so a future API or another interface cannot
    accidentally bypass the same rules used by the Streamlit screen.
    """
    clean = validate_record(record, system.input_ranges)
    if not units_confirmed:
        raise DataValidationError("Confirm the measurement values and units before predicting.")
    explanation = system.model.explain_one(clean)
    risk_number, risk_name = risk_for_label(explanation.prediction)
    history_id = history.add(
        model_id=system.model_id,
        input_source=input_source,
        predicted_label=explanation.prediction,
        risk_class=f"{risk_number} — {risk_name}",
        confidence=explanation.confidence,
        confidence_level=explanation.confidence_level,
        leaf_samples=explanation.leaf_samples,
        outside_support=explanation.outside_training_support,
        features=clean,
    )
    return SavedPrediction(explanation, clean, history_id)


@dataclass(frozen=True)
class SystemBundle:
    data: pd.DataFrame
    training_data: pd.DataFrame
    test_data: pd.DataFrame
    model: NumericalID3Classifier
    audit: DataAudit
    examples: dict[str, list[dict[str, float]]]
    input_ranges: dict[str, tuple[float, float]]
    model_id: str


def build_system(dataset_path: str | Path = DEFAULT_DATASET_PATH) -> SystemBundle:
    data = validate_training_data(load_dataset(dataset_path))
    train, test = historical_split(data)
    selection_path = DEFAULT_ARTIFACT_DIR / "latest" / "selected_model.json"
    configuration: dict[str, int | float] = {"max_depth": 10}
    selected_features = list(FEATURES)
    if selection_path.is_file():
        selection = json.loads(selection_path.read_text())
        configuration = selection.get("configuration", configuration)
        selected_features = selection.get("selected_features", selected_features)
    model = NumericalID3Classifier(**configuration).fit(train, selected_features, TARGET_COLUMN)
    audit = audit_dataset(data, dataset_path)
    config_tag = f"f{len(selected_features)}-d{configuration.get('max_depth', 10)}"
    model_id = f"id3-{config_tag}-{audit.dataset_sha256[:12]}"
    return SystemBundle(
        data=data,
        training_data=train,
        test_data=test,
        model=model,
        audit=audit,
        examples=build_example_records(train, examples_per_class=5),
        input_ranges={
            feature: (float(train[feature].min()), float(train[feature].max()))
            for feature in FEATURES
        },
        model_id=model_id,
    )
