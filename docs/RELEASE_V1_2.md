# Verified release 1.2.0

All 28 automated tests pass. Ruff and git whitespace checks pass.
Browser checks confirm computed red borders, hamburger open/close, all-zero reset,
highlighted diagnosis and mapped risk, no JSON path, and missing-field clearing on partial upload.
The documentation was rendered and all 10 pages visually reviewed.

Recomputed selected-model historical results: 300/303 correct (99.01% accuracy),
98.83% balanced accuracy, 96.14% macro-F1, 99.09% weighted-F1.
No new feature selection or hyperparameter tuning was performed for this UI revision.
These are dataset-specific benchmark results, not clinical validation.

Windows start: run_app.bat. macOS/Linux start: bash run_app.sh.
The first screen must say Version 1.2.0.
