
import math
import pandas as pd
from collections import Counter


def entropy(labels):
    """Calculate the entropy of a collection of class labels."""
    counts = Counter(labels)
    total = len(labels)

    if total == 0:
        return 0.0

    result = 0.0
    for count in counts.values():
        probability = count / total
        result -= probability * math.log2(probability)

    return result


def information_gain(data, feature, target):
    """Calculate information gain for a categorical feature."""
    parent_entropy = entropy(data[target])
    total = len(data)

    weighted_entropy = 0.0

    for value, subset in data.groupby(feature):
        weighted_entropy += (
            len(subset) / total
        ) * entropy(subset[target])

    return parent_entropy - weighted_entropy


def build_tree(data, features, target):
    """Build an ID3 decision tree recursively."""

    labels = data[target]

    # If all records belong to one class, return that class.
    if labels.nunique() == 1:
        return labels.iloc[0]

    # If no features remain, return the majority class.
    if not features:
        return labels.mode().iloc[0]

    # Select the feature with the highest information gain.
    gains = {
        feature: information_gain(data, feature, target)
        for feature in features
    }

    best_feature = max(gains, key=gains.get)
    tree = {"feature": best_feature, "branches": {}}

    remaining_features = [
        feature for feature in features
        if feature != best_feature
    ]

    for value, subset in data.groupby(best_feature):
        tree["branches"][value] = build_tree(
            subset,
            remaining_features,
            target
        )

    return tree


def predict(tree, record, default_class):
    """Classify one record using the constructed tree."""

    if not isinstance(tree, dict):
        return tree

    feature = tree["feature"]
    value = record.get(feature)

    # Handle unseen or missing feature values.
    if value not in tree["branches"]:
        return default_class

    return predict(
        tree["branches"][value],
        record,
        default_class
    )


def print_tree(tree, indent=""):
    """Display the decision tree in a readable format."""

    if not isinstance(tree, dict):
        print(indent + "Predict:", tree)
        return

    feature = tree["feature"]
    print(indent + "Feature:", feature)

    for value, subtree in tree["branches"].items():
        print(indent + f"  If {feature} = {value}:")
        print_tree(subtree, indent + "    ")


def main():
    # Illustrative categorical voting dataset.
    # This is a reconstructed demonstration dataset,
    # NOT the author's original dataset.
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
    features = [column for column in data.columns
                if column != target]

    print("=" * 50)
    print("STANDARD ID3 DECISION TREE DEMONSTRATION")
    print("=" * 50)

    print("\nDataset:")
    print(data.to_string(index=False))

    print("\nInitial entropy:",
          round(entropy(data[target]), 4))

    print("\nInformation gain:")
    for feature in features:
        print(
            f"{feature}:",
            round(information_gain(data, feature, target), 4)
        )

    default_class = data[target].mode().iloc[0]
    tree = build_tree(data, features, target)

    print("\nConstructed decision tree:")
    print_tree(tree)

    predictions = [
        predict(tree, row, default_class)
        for row in data[features].to_dict("records")
    ]

    actual = data[target].tolist()
    correct = sum(
        prediction == truth
        for prediction, truth in zip(predictions, actual)
    )

    accuracy = correct / len(actual) * 100

    print("\nPredictions:")
    for index, (prediction, truth) in enumerate(
        zip(predictions, actual), start=1
    ):
        print(
            f"Record {index}: predicted={prediction}, "
            f"actual={truth}"
        )

    print(f"\nTraining accuracy: {accuracy:.2f}%")
    print("Program completed successfully.")


if __name__ == "__main__":
    main()

