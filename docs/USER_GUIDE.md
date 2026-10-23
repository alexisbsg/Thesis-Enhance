# User guide version 1.2

Use the opening section and upload section of CODEBASE_AND_SOP_GUIDE.md for the complete guide.

1. Launch run_app.bat or run_app.sh and verify Version 1.2.0.
2. Load a preset or enter all 14 values. Uploads are reviewed before use.
3. Confirm values and units, then predict. Invalid entries show inline red feedback and do not save.
4. Read the highlighted diagnosis and mapped risk, component comparisons, confidence and text path.
5. Open the hamburger menu for Test history or Data quality.
6. Reset inputs to 0 clears the form and result; it does not delete past tests.

Risk classes are a dataset-label mapping, not validated clinical severity. Healthy medians are
comparison values, not clinical normal limits or targets to increase/decrease a component.

Plain CSV/TXT, selectable PDF (up to 10 pages), and images with Tesseract are supported.
Use samples/cbc_training_example.csv to try a complete training-only example. Missing fields from
partial uploads remain blank. No automatic unit conversion or arbitrary report-layout guarantee.
