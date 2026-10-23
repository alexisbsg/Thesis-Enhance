"""Readable numerical ID3 implementation with explainable nodes."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


def entropy(labels: Iterable[str]) -> float:
    """Return Shannon entropy in bits; an empty collection has zero entropy."""
    # SOP 2: one shared Shannon-entropy formula is used for parent and child nodes.
    counts = Counter(labels)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum((count / total) * math.log2(count / total) for count in counts.values())


@dataclass(frozen=True)
class SplitScore:
    feature: str
    threshold: float
    parent_entropy: float
    weighted_entropy: float
    information_gain: float
    left_count: int
    right_count: int


@dataclass
class TreeNode:
    prediction: str
    class_counts: dict[str, int]
    sample_count: int
    entropy: float
    depth: int
    stop_reason: str | None = None
    feature: str | None = None
    threshold: float | None = None
    information_gain: float | None = None
    left: TreeNode | None = None
    right: TreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature is None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PredictionExplanation:
    prediction: str
    confidence: float
    confidence_level: str
    leaf_samples: int
    class_counts: dict[str, int]
    decision_path: tuple[str, ...]
    outside_training_support: tuple[str, ...]


class NumericalID3Classifier:
    """Binary numerical ID3-style tree using entropy and information gain.

    Features remain eligible at deeper nodes because different thresholds of a
    continuous measurement may be useful in separate regions of the tree.
    """

    def __init__(
        self,
        *,
        max_depth: int = 10,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_information_gain: float = 0.0,
    ) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        if min_information_gain < 0:
            raise ValueError("min_information_gain cannot be negative")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.root_: TreeNode | None = None
        self.features_: tuple[str, ...] = ()
        self.classes_: tuple[str, ...] = ()
        self.feature_ranges_: dict[str, tuple[float, float]] = {}

    def fit(
        self, data: pd.DataFrame, features: Sequence[str], target: str
    ) -> NumericalID3Classifier:
        """Fit a tree; feature order defines deterministic tie-breaking."""
        if data.empty:
            raise ValueError("Cannot fit an empty dataset")
        self.features_ = tuple(features)
        self.classes_ = tuple(sorted(map(str, data[target].unique())))
        self.feature_ranges_ = {
            feature: (float(data[feature].min()), float(data[feature].max()))
            for feature in self.features_
        }
        self.root_ = self._build(data.reset_index(drop=True), target, depth=0)
        return self

    def _build(self, data: pd.DataFrame, target: str, depth: int) -> TreeNode:
        labels = data[target].astype(str)
        counts = {str(label): int(count) for label, count in labels.value_counts().items()}
        majority = labels.mode().iloc[0]
        node_entropy = entropy(labels)
        node = TreeNode(majority, counts, len(data), node_entropy, depth)
        # SOP 3: stop recursion on pure labels, depth, parent size, or insufficient gain.
        # The majority class remains the prediction when no admissible branch exists.
        if len(counts) == 1:
            node.stop_reason = "pure_node"
            return node
        if depth >= self.max_depth:
            node.stop_reason = "max_depth"
            return node
        if len(data) < self.min_samples_split:
            node.stop_reason = "min_samples_split"
            return node
        split = self.best_split(data, self.features_, target)
        if split is None:
            node.stop_reason = "no_valid_split"
            return node
        if split.information_gain <= self.min_information_gain:
            node.stop_reason = "min_information_gain"
            return node
        left_data = data[data[split.feature] <= split.threshold]
        right_data = data[data[split.feature] > split.threshold]
        node.feature = split.feature
        node.threshold = split.threshold
        node.information_gain = split.information_gain
        node.left = self._build(left_data, target, depth + 1)
        node.right = self._build(right_data, target, depth + 1)
        return node

    def score_candidates(
        self, data: pd.DataFrame, features: Sequence[str], target: str
    ) -> list[SplitScore]:
        """Score every valid midpoint split in deterministic order."""
        parent_entropy = entropy(data[target].astype(str))
        scores: list[SplitScore] = []
        total = len(data)
        for feature in features:
            values = np.sort(data[feature].dropna().unique().astype(float))
            if len(values) < 2:
                continue
            # SOP 2: midpoints enumerate numerical decisions without arbitrary bins.
            thresholds = (values[:-1] + values[1:]) / 2.0
            feature_values = data[feature].to_numpy(dtype=float)
            labels = data[target].astype(str).to_numpy()
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                left_count = int(left_mask.sum())
                right_count = total - left_count
                # SOP 3: reject this candidate if either child would be too small.
                # Other thresholds are still eligible; rejecting one is not a global stop.
                if min(left_count, right_count) < self.min_samples_leaf:
                    continue
                # SOP 2: weight child entropy by sample count before subtracting it.
                weighted = left_count / total * entropy(
                    labels[left_mask]
                ) + right_count / total * entropy(labels[~left_mask])
                scores.append(
                    SplitScore(
                        feature=feature,
                        threshold=float(threshold),
                        parent_entropy=parent_entropy,
                        weighted_entropy=weighted,
                        information_gain=parent_entropy - weighted,
                        left_count=left_count,
                        right_count=right_count,
                    )
                )
        return scores

    def best_split(
        self, data: pd.DataFrame, features: Sequence[str], target: str
    ) -> SplitScore | None:
        """Select the first highest-gain candidate, preserving baseline tie behavior."""
        best: SplitScore | None = None
        # SOP 2: strict > retains the first candidate on exact ties. Feature order
        # and ascending thresholds therefore make repeated fits deterministic.
        for candidate in self.score_candidates(data, features, target):
            if best is None or candidate.information_gain > best.information_gain:
                best = candidate
        return best

    def predict_one(self, record: dict[str, float]) -> str:
        return self.explain_one(record).prediction

    def predict(self, records: pd.DataFrame | list[dict[str, float]]) -> list[str]:
        rows = records.to_dict("records") if isinstance(records, pd.DataFrame) else records
        return [self.predict_one(row) for row in rows]

    def explain_one(self, record: dict[str, float]) -> PredictionExplanation:
        """Return the prediction, path, support warning, and smoothed leaf confidence."""
        if self.root_ is None:
            raise RuntimeError("The model has not been fitted")
        node = self.root_
        path: list[str] = []
        while not node.is_leaf:
            assert node.feature is not None and node.threshold is not None
            value = float(record[node.feature])
            goes_left = value <= node.threshold
            operator = "<=" if goes_left else ">"
            path.append(f"{node.feature} ({value:g}) {operator} {node.threshold:g}")
            next_node = node.left if goes_left else node.right
            if next_node is None:
                raise RuntimeError("Tree structure is incomplete")
            node = next_node
        predicted_count = node.class_counts.get(node.prediction, 0)
        # Laplace smoothing prevents a one-record pure leaf from reporting 100%.
        confidence = (predicted_count + 1) / (node.sample_count + len(self.classes_))
        outside = tuple(
            feature
            for feature, (minimum, maximum) in self.feature_ranges_.items()
            if not minimum <= float(record[feature]) <= maximum
        )
        if outside:
            confidence *= 0.75
        level = "High" if confidence >= 0.80 else "Moderate" if confidence >= 0.60 else "Low"
        return PredictionExplanation(
            prediction=node.prediction,
            confidence=float(confidence),
            confidence_level=level,
            leaf_samples=node.sample_count,
            class_counts=dict(node.class_counts),
            decision_path=tuple(path),
            outside_training_support=outside,
        )

    def iter_nodes(self) -> Iterable[TreeNode]:
        if self.root_ is None:
            return
        stack = [self.root_]
        while stack:
            node = stack.pop()
            yield node
            if node.right is not None:
                stack.append(node.right)
            if node.left is not None:
                stack.append(node.left)

    def complexity(self) -> dict[str, int]:
        nodes = list(self.iter_nodes())
        return {
            "internal_nodes": sum(not node.is_leaf for node in nodes),
            "leaf_nodes": sum(node.is_leaf for node in nodes),
            "max_depth": max((node.depth for node in nodes), default=0),
        }
