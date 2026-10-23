# Coding Rules

1. Use the shared package for all model and validation logic; do not copy it into the interface.
2. Keep functions focused and use descriptive names, type hints, and decision-focused docstrings.
3. Validate input at boundaries; never turn missing or invalid report values into zero.
4. Preserve deterministic feature order, threshold order, split seeds, and tie-breaking.
5. Fit feature selection and model settings only on training data.
6. Generate metrics from recorded runs. Never hard-code a desired score.
7. Preserve the baseline and add a regression test for every intentional behavior change.
8. Record data hashes, model configuration, selected features, and result artifacts.
9. Do not log uploaded report contents or commit runtime history, credentials, or private data.
10. Describe outputs as dataset labels and model confidence estimates, never medical diagnoses.
11. Keep risk assignment in one explicit mapping and fail closed when a new label lacks a mapping.
12. Never save a prediction when a required field is missing, non-finite, or outside the accepted
    training-partition range.
13. Present decision paths and leaf composition as readable tables, not raw dictionaries or JSON.

