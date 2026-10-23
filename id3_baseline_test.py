
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
        (len(subset) / total) * entropy(subset[target])
        for _, subset in data.groupby(feature)
    )

    return parent_entropy - weighted_entropy


def build_tree(data, features, target, default_class):
    labels = data[target]

    if labels.nunique() == 1:
        return labels.iloc[0]

    if not features or data.empty:
        return default_class

    gains = {
        feature: information_gain(data, feature, target)
        for feature in features
    }

    best_feature = max(gains, key=gains.get)
    tree = {"feature": best_feature, "branches": {}}

    remaining = [f for f in features if f != best_feature]

    for value, subset in data.groupby(best_feature):
        majority = subset[target].mode().iloc[0]
        tree["branches"][value] = build_tree(
            subset, remaining, target, majority
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
    # Illustrative data for testing the baseline only.
    # This is not the author's original voting dataset.
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

    # Hold out four records from tree construction.
    train = data.iloc[:8].copy()
    test = data.iloc[8:].copy()

    target = "Class"
    features = [c for c in data.columns if c != target]
    default_class = train[target].mode().iloc[0]

    print("STANDARD ID3 BASELINE — TRAIN/TEST DEMONSTRATION")
    print("\nTraining records:", len(train))
    print("Test records:", len(test))
    print("\nTraining entropy:", round(entropy(train[target]), 4))

    print("\nTraining information gain:")
    for feature in features:
        print(
            f"{feature}:",
            round(information_gain(train, feature, target), 4)
        )

    tree = build_tree(train, features, target, default_class)

    print("\nConstructed decision tree:")
    print_tree(tree)

    train_predictions = [
        predict(tree, row, default_class)
        for row in train[features].to_dict("records")
    ]

    test_predictions = [
        predict(tree, row, default_class)
        for row in test[features].to_dict("records")
    ]

    train_accuracy = sum(
        p == actual
        for p, actual in zip(train_predictions, train[target].tolist())
    ) / len(train) * 100

    test_accuracy = sum(
        p == actual
        for p, actual in zip(test_predictions, test[target].tolist())
    ) / len(test) * 100

    print("\nTest predictions:")
    for index, (prediction, actual) in enumerate(
        zip(test_predictions, test[target].tolist()), start=1
    ):
        print(f"Test record {index}: predicted={prediction}, actual={actual}")

    print(f"\nTraining accuracy: {train_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print("\nProgram completed successfully.")


if __name__ == "__main__":
    main()
