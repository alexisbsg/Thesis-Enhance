# SOP Methods and Solutions

## SOP 1 Feature Selection

Each CBC feature is evaluated independently across valid midpoint thresholds using training data.
The report records its best threshold, parent entropy, weighted child entropy, information gain,
and child sample sizes. Whole-tree usage is reported separately because a feature with weak root
gain can still help inside a branch.

Candidate sets of 8, 10, 12, and 14 features are recalculated in each cross-validation training
fold. This avoids using validation or test labels to rank features. The selected ten-feature set
matched the full model's held-out result while removing four unused/weaker candidates from model
fitting. The UI retains all columns because they remain part of the declared input schema and may
be needed by later model versions.

## SOP 2 Entropy and Node Selection

For every node, the model tests midpoint thresholds between adjacent distinct values. Splits with
children smaller than `min_samples_leaf` are rejected. The winning candidate has the highest
information gain; fixed feature and threshold order resolve exact ties deterministically.

Every node stores sample count, class counts, entropy, feature, threshold, gain, depth, prediction,
and stopping reason. Stability is evaluated across seeds 42, 7, 21, 100, and 2026. HGB at 12.95
remained the root in all five runs. This evidence does not support claiming that observed feature
overlap made the root unstable.

`min_samples_split` controls whether a parent is large enough to consider splitting.
`min_samples_leaf` controls the minimum records permitted in both resulting children. The selected
values are 6 and 3 respectively. For example, a six-record parent may split 3/3, but a 5/1 split is
rejected because the one-record leaf would be fragile.

## SOP 3 Recursive Branching and Sensitivity

The experiment compares maximum depth, minimum split size, minimum leaf size, and minimum gain
using the same training folds. The selected safeguards retained the original held-out metrics.
Feature reuse remains allowed because a continuous measurement may need different thresholds in
different branches.

Sensitivity is measured with a fixed fitted model. Each test measurement receives seeded positive
or negative perturbations equal to alpha times the feature standard deviation calculated from the
training data. Mean prediction change rates were 0.89% at alpha 0.005, 1.06% at alpha 0.01, and
6.73% at alpha 0.02. This is a model stress test, not a validated laboratory-error model.

