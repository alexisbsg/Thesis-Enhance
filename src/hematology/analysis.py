"""SOP 1, SOP 2, and SOP 3 analysis functions."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .constants import FEATURES, TARGET_COLUMN
from .data import historical_split
from .evaluation import calculate_metrics
from .models import NumericalID3Classifier


def feature_ranking(training_data: pd.DataFrame) -> pd.DataFrame:
    """SOP 1: best training-only single split for every feature."""
    model = NumericalID3Classifier()
    rows: list[dict[str, object]] = []
    # SOP 1: compare each feature on the same training subset; do not use test labels.
    for feature in FEATURES:
        split = model.best_split(training_data, [feature], TARGET_COLUMN)
        if split is not None:
            rows.append(vars(split))
    return (
        pd.DataFrame(rows)
        .sort_values(["information_gain", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )


def select_top_features(training_data: pd.DataFrame, top_k: int) -> list[str]:
    ranking = feature_ranking(training_data)
    return ranking["feature"].head(top_k).tolist()


def node_selection_report(model: NumericalID3Classifier) -> pd.DataFrame:
    """SOP 2: explain each selected node and every stopping reason."""
    rows = []
    for node_id, node in enumerate(model.iter_nodes()):
        rows.append(
            {
                "node_id": node_id,
                "depth": node.depth,
                "sample_count": node.sample_count,
                "entropy": node.entropy,
                "prediction": node.prediction,
                "feature": node.feature,
                "threshold": node.threshold,
                "information_gain": node.information_gain,
                "is_leaf": node.is_leaf,
                "stop_reason": node.stop_reason,
                "class_counts": str(node.class_counts),
            }
        )
    return pd.DataFrame(rows)


def root_stability(data: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    """SOP 2: measure root selection under specified stratified splits."""
    rows = []
    # SOP 2: changing training membership tests stability, not mathematical correctness.
    # This is a descriptive baseline analysis and never selects hyperparameters.
    for seed in seeds:
        train, _ = historical_split(data, seed=seed)
        split = NumericalID3Classifier().best_split(train, FEATURES, TARGET_COLUMN)
        if split is not None:
            rows.append({"seed": seed, **vars(split)})
    return pd.DataFrame(rows)


def feature_usage(model: NumericalID3Classifier) -> pd.DataFrame:
    # SOP 1: root ranking alone misses features useful only deeper in the tree.
    counts = Counter(node.feature for node in model.iter_nodes() if not node.is_leaf)
    return pd.DataFrame(
        [{"feature": feature, "split_count": counts.get(feature, 0)} for feature in FEATURES]
    ).sort_values(["split_count", "feature"], ascending=[False, True])


def sensitivity_analysis(
    model: NumericalID3Classifier,
    test_data: pd.DataFrame,
    training_data: pd.DataFrame,
    *,
    alphas: tuple[float, ...] = (0.005, 0.01, 0.02),
    repetitions: int = 10,
    seed: int = 2026,
) -> pd.DataFrame:
    """SOP 3: deterministic stress test using train-derived standard deviations."""
    generator = np.random.default_rng(seed)
    base = model.predict(test_data.loc[:, FEATURES])
    # SOP 3: use training-derived scale for perturbations; never fit it on test rows.
    standard_deviations = training_data.loc[:, FEATURES].std(ddof=0)
    rows: list[dict[str, float | int]] = []
    for alpha in alphas:
        for repetition in range(1, repetitions + 1):
            signs = generator.choice([-1.0, 1.0], size=(len(test_data), len(FEATURES)))
            perturbed = test_data.loc[:, FEATURES].copy()
            perturbation = signs * standard_deviations.to_numpy() * alpha
            perturbed.loc[:, FEATURES] = perturbed.to_numpy() + perturbation
            changed = sum(a != b for a, b in zip(base, model.predict(perturbed), strict=True))
            rows.append(
                {
                    "alpha": alpha,
                    "repetition": repetition,
                    "changed_predictions": changed,
                    "change_rate": changed / len(test_data),
                }
            )
    return pd.DataFrame(rows)


def evaluate_feature_subsets(
    training_data: pd.DataFrame,
    validation_data: pd.DataFrame,
    top_k_values: tuple[int, ...] = (8, 10, 12, 14),
) -> pd.DataFrame:
    """SOP 1 descriptive comparison on an explicitly provided validation split."""
    rows = []
    for top_k in top_k_values:
        selected = select_top_features(training_data, top_k)
        model = NumericalID3Classifier().fit(training_data, selected, TARGET_COLUMN)
        predicted = model.predict(validation_data.loc[:, selected])
        rows.append(
            {
                "top_k": top_k,
                "features": ",".join(selected),
                **calculate_metrics(validation_data[TARGET_COLUMN].tolist(), predicted).to_dict(),
                **model.complexity(),
            }
        )
    return pd.DataFrame(rows)
