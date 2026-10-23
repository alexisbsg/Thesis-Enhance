
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


def information_gain(data, feature, target, threshold):
    parent_entropy = entropy(data[target])
    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    if left.empty or right.empty:
        return 0.0

    weighted_entropy = (
        len(left) / len(data) * entropy(left[target])
        + len(right) / len(data) * entropy(right[target])
    )

    return parent_entropy - weighted_entropy


def best_split(data, features, target):
    best_split_info = None
    best_gain = 0.0

    for feature in features:
        values = sorted(data[feature].dropna().unique())

        for i in range(len(values) - 1):
            threshold = (values[i] + values[i + 1]) / 2
            gain = information_gain(data, feature, target, threshold)

            if gain > best_gain:
                best_gain = gain
                best_split_info = (feature, threshold)

    return best_split_info, best_gain


def build_tree(data, features, target, default_class, depth=0, max_depth=10):
    if data.empty:
        return default_class

    labels = data[target]

    if labels.nunique() == 1:
        return labels.iloc[0]

    majority = labels.mode().iloc[0]

    if not features or depth >= max_depth:
        return majority

    split, gain = best_split(data, features, target)

    if split is None or gain <= 0:
        return majority

    feature, threshold = split
    left = data[data[feature] <= threshold]
    right = data[data[feature] > threshold]

    return {
        "feature": feature,
        "threshold": threshold,
        "default": majority,
        "left": build_tree(
            left, features, target, majority, depth + 1, max_depth
        ),
        "right": build_tree(
            right, features, target, majority, depth + 1, max_depth
        )
    }


def predict(tree, record, default_class):
    while isinstance(tree, dict):
        value = record[tree["feature"]]

        if pd.isna(value):
            return tree["default"]

        if value <= tree["threshold"]:
            tree = tree["left"]
        else:
            tree = tree["right"]

    return tree


def main():
    file_path = "diagnosed_cbc_screened.csv"

    try:
        data = pd.read_csv(file_path)
    except Exception as error:
        print("Could not load dataset:", error)
        return

    data.columns = data.columns.astype(str).str.strip()
    data = data.dropna(how="all").reset_index(drop=True)

    print("\nColumns:", data.columns.tolist())
    print("Rows:", len(data))

    target = input("Enter target column name: ").strip()

    if target not in data.columns:
        print("Target column not found.")
        return

    data[target] = data[target].astype("string").str.strip()
    data = data.dropna(subset=[target]).reset_index(drop=True)
    data[target] = data[target].astype(str)

    features = [column for column in data.columns if column != target]

    # Convert feature columns to numeric.
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")

    # Drop rows with missing or invalid feature values for this baseline.
    data = data.dropna(subset=features).reset_index(drop=True)

    if data.empty or data[target].nunique() < 2:
        print("Insufficient usable records or target classes.")
        return

    print("\nTarget classes:", data[target].value_counts().to_dict())
    print("Complete usable records:", len(data))

    try:
        train, test = train_test_split(
            data,
            test_size=0.25,
            random_state=42,
            stratify=data[target]
        )
    except ValueError as error:
        print("Could not split dataset:", error)
        return

    default_class = train[target].mode().iloc[0]
    tree = build_tree(train, features, target, default_class)

    predictions = [
        predict(tree, row, default_class)
        for row in test[features].to_dict("records")
    ]

    print("\n--- NUMERICAL ID3 BASELINE ---")
    print("Training records:", len(train))
    print("Testing records:", len(test))
    print(f"Accuracy: {accuracy_score(test[target], predictions) * 100:.2f}%")

    print("\nIndividual predictions:")
    for i, (actual, predicted) in enumerate(
        zip(test[target], predictions), start=1
    ):
        print(f"{i}. Actual={actual}, Predicted={predicted}")

    print("\nClassification report:")
    print(classification_report(
        test[target], predictions, zero_division=0
    ))


if __name__ == "__main__":
    main()
