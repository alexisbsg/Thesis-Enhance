"""Run the reproducible baseline, SOP reports, and training-only model comparison."""

from __future__ import annotations

import json

import pandas as pd

from hematology.analysis import (
    feature_ranking,
    feature_usage,
    node_selection_report,
    root_stability,
    select_top_features,
    sensitivity_analysis,
)
from hematology.constants import DEFAULT_ARTIFACT_DIR, DEFAULT_DATASET_PATH, TARGET_COLUMN
from hematology.data import audit_dataset, historical_split, load_dataset, validate_training_data
from hematology.evaluation import cross_validate_configurations, detailed_evaluation
from hematology.models import NumericalID3Classifier


def main() -> None:
    output = DEFAULT_ARTIFACT_DIR / "latest"
    output.mkdir(parents=True, exist_ok=True)
    data = validate_training_data(load_dataset(DEFAULT_DATASET_PATH))
    development, test = historical_split(data)
    audit = audit_dataset(data, DEFAULT_DATASET_PATH)
    configurations: list[dict[str, int | float]] = [
        {
            "max_depth": 10,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_information_gain": 0.0,
        },
        {
            "max_depth": 5,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "min_information_gain": 0.0,
        },
        {
            "max_depth": 10,
            "min_samples_split": 6,
            "min_samples_leaf": 3,
            "min_information_gain": 0.0,
        },
        {
            "max_depth": 10,
            "min_samples_split": 10,
            "min_samples_leaf": 5,
            "min_information_gain": 0.001,
        },
    ]
    candidate_results = []
    for feature_count in (8, 10, 12, 14):

        def selector(fold_data, count=feature_count):
            return select_top_features(fold_data, count)

        result = cross_validate_configurations(
            development,
            configurations,
            n_splits=3,
            n_repeats=1,
            seed=42,
            feature_selector=selector,
        )
        result["requested_feature_count"] = feature_count
        result["candidate_id"] = [
            f"f{feature_count}-c{configuration_id}"
            for configuration_id in result["configuration_id"]
        ]
        candidate_results.append(result)
    cv = pd.concat(candidate_results, ignore_index=True)
    summary = (
        cv.groupby(
            ["candidate_id", "configuration_id", "requested_feature_count"],
            as_index=False,
        )
        .agg(
            mean_macro_f1=("macro_f1", "mean"),
            std_macro_f1=("macro_f1", "std"),
            mean_balanced_accuracy=("balanced_accuracy", "mean"),
            mean_leaf_nodes=("leaf_nodes", "mean"),
        )
        .fillna(0)
    )
    # SOP 1 and SOP 3: select by training CV macro-F1, then prefer fewer features
    # and fewer leaves on a tie. Held-out test scores do not choose the winner.
    selected_row = summary.sort_values(
        ["mean_macro_f1", "requested_feature_count", "mean_leaf_nodes"],
        ascending=[False, True, True],
    ).iloc[0]
    selected_id = int(selected_row["configuration_id"])
    selected_feature_count = int(selected_row["requested_feature_count"])
    selected_features = select_top_features(development, selected_feature_count)
    selected_config = configurations[selected_id]
    model = NumericalID3Classifier(**selected_config).fit(
        development, selected_features, TARGET_COLUMN
    )
    predictions = model.predict(test.loc[:, selected_features])
    evaluation = detailed_evaluation(test[TARGET_COLUMN].tolist(), predictions)

    feature_ranking(development).to_csv(output / "sop1_feature_ranking.csv", index=False)
    feature_usage(model).to_csv(output / "sop1_feature_usage.csv", index=False)
    root_stability(data, [42, 7, 21, 100, 2026]).to_csv(
        output / "sop2_root_stability.csv", index=False
    )
    node_selection_report(model).to_csv(output / "sop2_selected_nodes.csv", index=False)
    sensitivity_analysis(model, test, development).to_csv(
        output / "sop3_sensitivity.csv", index=False
    )
    confidence_rows = []
    for (_, record), actual in zip(test.iterrows(), test[TARGET_COLUMN], strict=True):
        explanation = model.explain_one(record.to_dict())
        confidence_rows.append(
            {
                "actual": actual,
                "predicted": explanation.prediction,
                "correct": actual == explanation.prediction,
                "confidence": explanation.confidence,
                "confidence_level": explanation.confidence_level,
                "leaf_samples": explanation.leaf_samples,
            }
        )
    pd.DataFrame(confidence_rows).to_csv(output / "confidence_reliability.csv", index=False)
    cv.to_csv(output / "model_selection_folds.csv", index=False)
    summary.to_csv(output / "model_selection_summary.csv", index=False)
    (output / "selected_model.json").write_text(
        json.dumps(
            {
                "selection_metric": "mean cross-validation macro_f1",
                "selected_configuration_id": selected_id,
                "configuration": selected_config,
                "selected_feature_count": selected_feature_count,
                "selected_features": selected_features,
                "complexity": model.complexity(),
                "held_out_evaluation": evaluation,
                "data_audit": audit.to_dict(),
            },
            indent=2,
        )
    )
    print(
        json.dumps({"selected_configuration": selected_config, **evaluation["summary"]}, indent=2)
    )


if __name__ == "__main__":
    main()
