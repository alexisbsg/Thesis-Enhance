
import math
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def entropy(labels):
    counts = Counter(labels)
    total = len(labels)

    if total == 0:
        return 0.0

    return -sum(
        (count / total) * math.log2(count / total)
        for count in counts.values()
    )


def information_gain(data, feature, target):
    parent_entropy = entropy(data[target])
    total = len(data)

    if total == 0:
        return 0.0

    weighted_entropy = sum(
        (len(subset) / total) * entropy(subset[target])
        for _, subset in data.groupby(feature, dropna=False)
    )

    return parent_entropy - weighted_entropy


def build_tree(data, features, target, default_class):
    if data.empty:
        return default_class

    labels = data[target]

    if labels.nunique() == 1:
        return labels.iloc[0]

    if not features:
        return labels.mode().iloc[0]

    gains = {
        feature: information_gain(data, feature, target)
        for feature in features
    }

    best_feature = max(gains, key=gains.get)

    if gains[best_feature] <= 0:
        return labels.mode().iloc[0]

    tree = {"feature": best_feature, "branches": {}}
    remaining = [f for f in features if f != best_feature]

    for value, subset in data.groupby(best_feature, dropna=False):
        tree["branches"][value] = build_tree(
            subset, remaining, target, labels.mode().iloc[0]
        )

    return tree


def predict(tree, record, default_class):
    while isinstance(tree, dict):
        feature = tree["feature"]
        value = record.get(feature)

        if pd.isna(value) or value not in tree["branches"]:
            return default_class

        tree = tree["branches"][value]

    return tree


def main():
    file_path = input("CSV file path: ").strip().strip('"')

    try:
        data = pd.read_csv(file_path)
    except Exception as error:
        print("Could not load dataset:", error)
        return

    data.columns = data.columns.astype(str).str.strip()
    data = data.dropna(how="all").dropna().reset_index(drop=True)

    if data.empty:
        print("Dataset is empty or contains no complete records.")
        return

    print("\nAvailable columns:")
    for column in data.columns:
        print("-", column)

    target = input("\nEnter the target column name exactly: ").strip()

    if target not in data.columns:
        print("Error: target column not found.")
        return

    if data[target].nunique() < 2:
        print("Error: target must contain at least two classes.")
        return

    features = [column for column in data.columns if column != target]

    # Preliminary categorical version: numeric values are treated as categories.
    data[features] = data[features].astype(str)
    data[target] = data[target].astype(str)

    if len(data) < 5:
        print("Error: at least five complete records are needed.")
        return

    try:
        train, test = train_test_split(
            data,
            test_size=0.25,
            random_state=42,
            stratify=data[target]
        )
    except ValueError as error:
        print("Could not create a stratified train/test split:", error)
        print("Check whether each target class has enough records.")
        return

    default_class = train[target].mode().iloc[0]
    tree = build_tree(train, features, target, default_class)

    predictions = [
        predict(tree, row, default_class)
        for row in test[features].to_dict("records")
    ]

    print("\nIndividual test predictions:")
    for index, (actual, predicted) in enumerate(
        zip(test[target].tolist(), predictions), start=1
    ):
        print(
            f"Record {index}: "
            f"Actual={actual}, Predicted={predicted}"
    )
    print("\n--- ID3 BASELINE EVALUATION ---")
    print("Training records:", len(train))
    print("Testing records:", len(test))
    print("Target:", target)
    print("Target classes:", sorted(data[target].unique()))
    print("Test accuracy:",
          f"{accuracy_score(test[target], predictions) * 100:.2f}%")

    print("\nClassification report:")
    print(classification_report(
        test[target], predictions, zero_division=0
    ))


if __name__ == "__main__":
    main()
