"""Local SQLite prediction-history persistence with no patient identifiers."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from .interpretation import risk_for_label


class PredictionHistory:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS prediction_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    input_source TEXT NOT NULL,
                    predicted_label TEXT NOT NULL,
                    risk_class TEXT NOT NULL DEFAULT 'Unknown',
                    confidence REAL NOT NULL,
                    confidence_level TEXT NOT NULL,
                    leaf_samples INTEGER NOT NULL,
                    outside_support TEXT NOT NULL,
                    features_json TEXT NOT NULL
                )
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(prediction_history)").fetchall()
            }
            if "risk_class" not in columns:
                connection.execute(
                    "ALTER TABLE prediction_history "
                    "ADD COLUMN risk_class TEXT NOT NULL DEFAULT 'Unknown'"
                )
            missing_risk = connection.execute(
                "SELECT id, predicted_label FROM prediction_history WHERE risk_class = 'Unknown'"
            ).fetchall()
            for row in missing_risk:
                try:
                    risk_number, risk_name = risk_for_label(row["predicted_label"])
                except ValueError:
                    continue
                connection.execute(
                    "UPDATE prediction_history SET risk_class = ? WHERE id = ?",
                    (f"{risk_number} — {risk_name}", row["id"]),
                )

    def add(
        self,
        *,
        model_id: str,
        input_source: str,
        predicted_label: str,
        risk_class: str,
        confidence: float,
        confidence_level: str,
        leaf_samples: int,
        outside_support: tuple[str, ...],
        features: dict[str, float],
    ) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO prediction_history (
                    created_at, model_id, input_source, predicted_label,
                    risk_class, confidence, confidence_level, leaf_samples,
                    outside_support, features_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now(UTC).isoformat(),
                    model_id,
                    input_source,
                    predicted_label,
                    risk_class,
                    confidence,
                    confidence_level,
                    leaf_samples,
                    json.dumps(outside_support),
                    json.dumps(features, sort_keys=True),
                ),
            )
            return int(cursor.lastrowid)

    def list(self, limit: int = 500) -> pd.DataFrame:
        with self._connect() as connection:
            records = pd.read_sql_query(
                """
                SELECT id, created_at, model_id, input_source, predicted_label,
                       risk_class, confidence, confidence_level, leaf_samples, outside_support,
                       features_json
                FROM prediction_history ORDER BY id DESC LIMIT ?
                """,
                connection,
                params=(limit,),
            )
        if records.empty:
            return records
        feature_values = records.pop("features_json").map(json.loads).apply(pd.Series)
        return pd.concat([records, feature_values], axis=1)

    def clear(self) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM prediction_history")
