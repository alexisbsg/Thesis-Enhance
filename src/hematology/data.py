"""Dataset loading, validation, auditing, and deterministic splitting."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .constants import DEFAULT_SEED, FEATURES, TARGET_COLUMN, TEST_SIZE


class DataValidationError(ValueError):
    """Raised when data cannot safely enter the model pipeline."""


@dataclass(frozen=True)
class DataAudit:
    row_count: int
    feature_count: int
    class_count: int
    missing_cells: int
    nonfinite_cells: int
    duplicate_rows: int
    conflicting_feature_groups: int
    dataset_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return asdict(self)


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load the required CBC schema and fail with an actionable message."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise DataValidationError(f"Dataset not found: {dataset_path}")
    data = pd.read_csv(dataset_path)
    data.columns = data.columns.astype(str).str.strip()
    missing = [column for column in (*FEATURES, TARGET_COLUMN) if column not in data]
    if missing:
        raise DataValidationError(f"Missing required columns: {missing}")
    data = data.loc[:, [*FEATURES, TARGET_COLUMN]].copy()
    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype("string").str.strip()
    return data


def validate_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Return a validated copy; never silently discard invalid model rows."""
    if data.empty:
        raise DataValidationError("The dataset is empty.")
    invalid_numeric = data.loc[:, FEATURES].isna().any(axis=1)
    nonfinite = ~np.isfinite(data.loc[:, FEATURES].to_numpy(dtype=float)).all(axis=1)
    missing_target = data[TARGET_COLUMN].isna() | data[TARGET_COLUMN].eq("")
    invalid_count = int((invalid_numeric | nonfinite | missing_target).sum())
    if invalid_count:
        raise DataValidationError(
            f"Dataset contains {invalid_count} rows with missing or non-finite required values."
        )
    if data[TARGET_COLUMN].nunique() < 2:
        raise DataValidationError("At least two target classes are required.")
    return data.reset_index(drop=True)


def audit_dataset(data: pd.DataFrame, source_path: str | Path) -> DataAudit:
    """Compute reproducible data-quality facts without changing the records."""
    feature_groups = data.groupby(list(FEATURES), dropna=False)[TARGET_COLUMN].nunique()
    raw_bytes = Path(source_path).read_bytes()
    values = data.loc[:, FEATURES].to_numpy(dtype=float)
    return DataAudit(
        row_count=len(data),
        feature_count=len(FEATURES),
        class_count=int(data[TARGET_COLUMN].nunique(dropna=True)),
        missing_cells=int(data.isna().sum().sum()),
        nonfinite_cells=int((~np.isfinite(values)).sum()),
        duplicate_rows=int(data.duplicated().sum()),
        conflicting_feature_groups=int((feature_groups > 1).sum()),
        dataset_sha256=hashlib.sha256(raw_bytes).hexdigest(),
    )


def historical_split(
    data: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the documented stratified historical train/test split."""
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=seed,
        stratify=data[TARGET_COLUMN],
    )
    return train.copy(), test.copy()


def record_validation_errors(
    record: dict[str, object],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, str]:
    """Return a field-to-message mapping suitable for inline form feedback."""
    errors: dict[str, str] = {}
    missing = [feature for feature in FEATURES if feature not in record]
    errors.update({feature: "A value is required." for feature in missing})
    for feature in FEATURES:
        if feature in missing:
            continue
        try:
            value = float(record[feature])
        except (TypeError, ValueError):
            errors[feature] = "Enter a numeric value."
            continue
        if not np.isfinite(value):
            errors[feature] = "Enter a finite numeric value."
            continue
        if ranges and feature in ranges:
            minimum, maximum = ranges[feature]
            if not minimum <= value <= maximum:
                errors[feature] = f"Must be between {minimum:g} and {maximum:g}."
        elif value < 0:
            errors[feature] = "Value cannot be negative."
    return errors


def validate_record(
    record: dict[str, object],
    ranges: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Validate a prediction record and return features in canonical order."""
    errors = record_validation_errors(record, ranges)
    if errors:
        detail = "; ".join(f"{feature}: {message}" for feature, message in errors.items())
        raise DataValidationError(detail)
    clean: dict[str, float] = {}
    for feature in FEATURES:
        value = float(record[feature])
        clean[feature] = value
    return clean
