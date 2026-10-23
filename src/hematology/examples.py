"""Training-only example records for quick UI autofill."""

from __future__ import annotations

import pandas as pd

from .constants import FEATURES, TARGET_COLUMN


def build_example_records(
    training_data: pd.DataFrame, examples_per_class: int = 5
) -> dict[str, list[dict[str, float]]]:
    """Select deterministic representative records nearest each class median."""
    examples: dict[str, list[dict[str, float]]] = {}
    for label, group in training_data.groupby(TARGET_COLUMN, sort=True):
        median = group.loc[:, FEATURES].median()
        ranges = group.loc[:, FEATURES].max() - group.loc[:, FEATURES].min()
        safe_ranges = ranges.replace(0, 1)
        distance = ((group.loc[:, FEATURES] - median) / safe_ranges).abs().sum(axis=1)
        selected = group.loc[distance.sort_values(kind="stable").index].head(examples_per_class)
        examples[str(label)] = [
            {feature: float(row[feature]) for feature in FEATURES} for _, row in selected.iterrows()
        ]
    return examples
