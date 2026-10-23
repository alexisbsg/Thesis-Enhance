# Architecture

The application follows one-way dependencies: the Streamlit interface calls `service.py`; the
service uses validated data, the shared model, examples, and saved experiment selection. Analysis
and evaluation modules call the same shared model. No model mathematics is duplicated in the UI.

## Prediction flow

1. Load and validate the declared dataset schema.
2. Recreate the historical stratified split with seed 42.
3. Load the training-selected feature/configuration record when available.
4. Fit the numerical ID3 model on training records only.
5. Accept manual, example, or reviewed report input.
6. Validate all 14 required measurements against training-partition ranges and require unit review.
7. Block prediction and persistence when any field fails validation.
8. Traverse the tree and capture every feature-threshold decision.
9. Calculate the smoothed leaf confidence, mapped risk class, and component comparison.
10. Display and save the valid result locally without patient identity.

## Trust boundaries

Uploaded files stay in memory. The system does not log report text. The history database stores
measurements and research results, so it must still be protected as potentially sensitive data.
The runtime database and generated artifacts are excluded from version control.

## Model selection boundary

All feature and pruning choices are evaluated inside the training partition. The held-out test set
is used only after selection. The historical test has already been examined, so it is a reproducible
benchmark rather than fresh external evidence.

