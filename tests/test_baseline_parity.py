import pandas as pd
from sklearn.model_selection import train_test_split

from hematology.constants import FEATURES, TARGET_COLUMN
from hematology.models import NumericalID3Classifier


def test_baseline_matches_reported_root_and_accuracy():
    data = pd.read_csv("diagnosed_cbc_screened.csv")
    train, test = train_test_split(
        data, test_size=0.25, random_state=42, stratify=data[TARGET_COLUMN]
    )
    model = NumericalID3Classifier(max_depth=10).fit(train, FEATURES, TARGET_COLUMN)
    assert model.root_ is not None
    assert model.root_.feature == "HGB"
    assert model.root_.threshold == 12.95
    predictions = model.predict(test.loc[:, FEATURES])
    accuracy = sum(a == b for a, b in zip(test[TARGET_COLUMN], predictions, strict=True)) / len(
        test
    )
    assert accuracy == 300 / 303
