import pandas as pd
import pytest

from hematology.models import NumericalID3Classifier, entropy


def test_entropy_boundaries():
    assert entropy([]) == 0.0
    assert entropy(["A", "A"]) == 0.0
    assert entropy(["A", "B"]) == pytest.approx(1.0)


def test_minimum_leaf_size_rejects_small_child():
    data = pd.DataFrame({"x": [0, 1, 2, 3], "label": ["A", "A", "A", "B"]})
    model = NumericalID3Classifier(min_samples_leaf=3).fit(data, ["x"], "label")
    assert model.root_ is not None
    assert model.root_.is_leaf
    assert model.root_.stop_reason == "no_valid_split"


def test_explanation_uses_laplace_smoothing():
    data = pd.DataFrame({"x": [0, 1, 2, 3], "label": ["A", "A", "B", "B"]})
    model = NumericalID3Classifier().fit(data, ["x"], "label")
    result = model.explain_one({"x": 0})
    assert result.prediction == "A"
    assert 0 < result.confidence < 1
