
import math
import pandas as pd
from collections import Counter


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

    weighted_entropy = sum(
        len(subset) / total * entropy(subset[target])
        for _, subset in data.groupby(feature)
    )

    return parent_entropy - weighted_entropy


def build_tree(data, features, target):
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
    remaining = [f for f in features if f != best_feature]

    tree = {"feature": best_feature, "branches": {}}

    for value, subset in data.groupby(best_feature):
        tree["branches"][value] = build_tree(
            subset, remaining, target
        )

    return tree


def predict(tree, record, default_class):
    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    value = record.get(feature)

    if value not in tree["branches"]:
        return default_class

    return predict(tree["branches"][value], record, default_class)


def print_tree(tree, indent=""):
    if not isinstance(tree, dict):
        print(indent + "Predict:", tree)
        return

    feature = tree["feature"]
    print(indent + "Feature:", feature)

    for value, subtree in tree["branches"].items():
        print(indent + f"  If {feature} = {value}:")
        print_tree(subtree, indent + "    ")


def main():
    # Illustrative voting-style dataset for software testing only.
    data = pd.DataFrame([
        ["yes", "yes", "yes", "A"],
        ["yes", "yes", "no",  "A"],
        ["yes", "no",  "yes", "A"],
        ["no",  "yes", "yes", "B"],
        ["no",  "no",  "yes", "B"],
        ["no",  "no",  "no",  "B"],
        ["yes", "no",  "no",  "B"],
        ["no",  "yes", "no",  "B"],
        ["yes", "yes", "yes", "A"],
        ["no",  "yes", "yes", "B"],
        ["yes", "no",  "yes", "A"],
        ["no",  "no",  "yes", "B"],
    ], columns=["Policy_X", "Policy_Y", "Policy_Z", "Class"])

    target = "Class"
    features = [c for c in data.columns if c != target]
    default_class = data[target].mode().iloc[0]

    tree = build_tree(data, features, target)

    print("=" * 45)
    print("STANDARD ID3 — INTERACTIVE BASELINE")
    print("=" * 45)
    print("\nTrained Decision Tree:")
    print_tree(tree)

    print("\nEnter values for a new record.")
    print("Allowed values: yes or no")

    record = {}

    for feature in features:
        while True:
            value = input(f"{feature} (yes/no): ").strip().lower()

            if value in ("yes", "no"):
                record[feature] = value
                break

            print("Invalid input. Please enter yes or no.")

    result = predict(tree, record, default_class)

    print("\nInput record:", record)
    print("Predicted class:", result)
    print("\nThis is an illustrative software demonstration,")
    print("not a medical classification or clinical result.")


if __name__ == "__main__":
    main()
