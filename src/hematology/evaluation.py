"""Model metrics, cross-validation, and honest model comparisons."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold

from .constants import FEATURES, TARGET_COLUMN
from .models import NumericalID3Classifier


@dataclass(frozen=True)
class MetricSummary:
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def calculate_metrics(actual: list[str], predicted: list[str]) -> MetricSummary:
    return MetricSummary(
        accuracy=float(accuracy_score(actual, predicted)),
        balanced_accuracy=float(balanced_accuracy_score(actual, predicted)),
        macro_f1=float(f1_score(actual, predicted, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(actual, predicted, average="weighted", zero_division=0)),
    )


def detailed_evaluation(actual: list[str], predicted: list[str]) -> dict[str, object]:
    labels = sorted(set(actual) | set(predicted))
    return {
        "summary": calculate_metrics(actual, predicted).to_dict(),
        "labels": labels,
        "confusion_matrix": confusion_matrix(actual, predicted, labels=labels).tolist(),
        "classification_report": classification_report(
            actual, predicted, labels=labels, output_dict=True, zero_division=0
        ),
    }


def cross_validate_configurations(
    development_data: pd.DataFrame,
    configurations: list[dict[str, int | float]],
    *,
    n_splits: int = 3,
    n_repeats: int = 3,
    seed: int = 42,
    feature_selector: Callable[[pd.DataFrame], list[str]] | None = None,
) -> pd.DataFrame:
    """Compare configurations using fold-local feature selection and fitting."""
    splitter = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    rows: list[dict[str, object]] = []
    y = development_data[TARGET_COLUMN]
    for fold, (train_index, validation_index) in enumerate(splitter.split(development_data, y), 1):
        fold_train = development_data.iloc[train_index]
        fold_validation = development_data.iloc[validation_index]
        # SOP 1: refit feature ranking inside each fold to prevent validation leakage.
        # SOP 3: all tree-control candidates share these folds for a fair comparison.
        selected_features = feature_selector(fold_train) if feature_selector else list(FEATURES)
        for config_index, configuration in enumerate(configurations):
            model = NumericalID3Classifier(**configuration).fit(
                fold_train, selected_features, TARGET_COLUMN
            )
            predicted = model.predict(fold_validation.loc[:, selected_features])
            metrics = calculate_metrics(fold_validation[TARGET_COLUMN].tolist(), predicted)
            rows.append(
                {
                    "fold": fold,
                    "configuration_id": config_index,
                    **configuration,
                    "feature_count": len(selected_features),
                    **metrics.to_dict(),
                    **model.complexity(),
                }
            )
    return pd.DataFrame(rows)
