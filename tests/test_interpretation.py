import pytest

from hematology.constants import ANEMIA_LABELS, HIGH_RISK_LABELS
from hematology.interpretation import risk_for_label


def test_all_dataset_labels_have_requested_risk_mapping():
    assert risk_for_label("Healthy") == ("Class 1", "Low Risk")
    for label in ANEMIA_LABELS:
        assert risk_for_label(label) == ("Class 2", "Moderate Risk")
    for label in HIGH_RISK_LABELS:
        assert risk_for_label(label) == ("Class 3", "High Risk")


def test_unknown_dataset_label_fails_closed():
    with pytest.raises(ValueError, match="No risk mapping"):
        risk_for_label("Unknown")
