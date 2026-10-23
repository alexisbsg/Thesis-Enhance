import sqlite3

from hematology.history import PredictionHistory


def test_history_round_trip(tmp_path):
    history = PredictionHistory(tmp_path / "history.sqlite3")
    record_id = history.add(
        model_id="model-1",
        input_source="manual",
        predicted_label="Healthy",
        risk_class="Class 1 — Low Risk",
        confidence=0.8,
        confidence_level="High",
        leaf_samples=10,
        outside_support=(),
        features={"WBC": 5.0},
    )
    assert record_id == 1
    assert history.list().iloc[0]["predicted_label"] == "Healthy"
    assert history.list().iloc[0]["risk_class"] == "Class 1 — Low Risk"
    history.clear()
    assert history.list().empty


def test_existing_history_is_migrated_with_risk_class(tmp_path):
    path = tmp_path / "old_history.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                model_id TEXT NOT NULL,
                input_source TEXT NOT NULL,
                predicted_label TEXT NOT NULL,
                confidence REAL NOT NULL,
                confidence_level TEXT NOT NULL,
                leaf_samples INTEGER NOT NULL,
                outside_support TEXT NOT NULL,
                features_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO prediction_history (
                created_at, model_id, input_source, predicted_label, confidence,
                confidence_level, leaf_samples, outside_support, features_json
            ) VALUES ('now', 'old', 'manual', 'Leukemia', 0.8, 'High', 10, '[]', '{}')
            """
        )
    migrated = PredictionHistory(path).list()
    assert migrated.iloc[0]["risk_class"] == "Class 3 — High Risk"
