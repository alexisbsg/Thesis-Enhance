import pandas as pd
import pytest

from hematology.constants import FEATURES, TARGET_COLUMN
from hematology.data import (
    DataValidationError,
    record_validation_errors,
    validate_record,
    validate_training_data,
)


def test_record_requires_all_features():
    with pytest.raises(DataValidationError, match="A value is required"):
        validate_record({"WBC": 1.0})


def test_training_validation_does_not_silently_drop_rows():
    row = {feature: 1.0 for feature in FEATURES}
    valid = pd.DataFrame([{**row, TARGET_COLUMN: "A"}, {**row, "WBC": 2.0, TARGET_COLUMN: "B"}])
    invalid = valid.copy()
    invalid.loc[0, "WBC"] = None
    with pytest.raises(DataValidationError, match="1 rows"):
        validate_training_data(invalid)


def test_record_range_errors_are_reported_per_field():
    record = {feature: 5.0 for feature in FEATURES}
    ranges = {feature: (1.0, 10.0) for feature in FEATURES}
    record["WBC"] = 11.0
    record["HGB"] = 0.0
    errors = record_validation_errors(record, ranges)
    assert errors == {
        "WBC": "Must be between 1 and 10.",
        "HGB": "Must be between 1 and 10.",
    }
    with pytest.raises(DataValidationError, match="WBC"):
        validate_record(record, ranges)
