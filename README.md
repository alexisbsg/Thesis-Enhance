# ID3 Hematological Research MVP version 1.2.0

This project is an explainable numerical ID3-style classifier for CBC records. It includes a
Streamlit interface, deterministic training, held-out evaluation, reports for all three research
SOPs, prediction history, five training-only example records per dataset class, report-text
extraction, and a transparent confidence estimate.

The system is an academic research prototype. It predicts labels contained in the supplied
dataset. It is not clinically validated and must not be used for diagnosis or treatment decisions.

## Verified result

The included experiment selected ten features with `min_samples_split=6`,
`min_samples_leaf=3`, and `max_depth=10` using three-fold cross-validation within the historical
training partition. On the previously examined historical test partition it reproduced:

- Accuracy: 99.01%
- Balanced accuracy: 98.83%
- Macro-F1: 96.14%
- Weighted-F1: 99.09%
- Tree: 14 internal nodes, 15 leaves, actual depth 5

These results describe one screened dataset and split. Dataset provenance, authoritative units,
label-generation method, license, patient independence, and external clinical validity remain to
be confirmed.

## Open the corrected build

Stop the old Streamlit process, extract this archive into a new folder, and run
`run_app.bat` on Windows or `bash run_app.sh` on macOS/Linux. Verify **Version 1.2.0**
at the top of the page. The hamburger button is at the top left. The launchers install
this folder's package and run this folder's `app.py`.

The full code walkthrough and SOP explanations are in
`docs/ID3_Hematological_Complete_Documentation.docx` and `docs/CODEBASE_AND_SOP_GUIDE.md`.
Search the Python source for `SOP 1`, `SOP 2`, and `SOP 3` to find the annotated logic.

## Install and run

Use Python 3.11 or newer.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e .
python -m streamlit run app.py
```

Image OCR also requires the Tesseract application. Manual input, examples, and text-based PDF
extraction do not require Tesseract.

## Verify and reproduce

```bash
python -m pip install -e ".[dev]"
python -m pytest
python -m ruff check app.py src scripts tests
python scripts/audit_data.py
python scripts/run_experiments.py
```

The experiment command writes CSV and JSON reports to `artifacts/latest/`. Model selection uses
only cross-validation folds inside the training partition. The test partition is evaluated after a
configuration is selected.

## Features

- Manual entry for all 14 CBC columns.
- Five representative existing records for each of the nine dataset classes. Examples are selected
  deterministically from the training partition and autofill every CBC column.
- Conservative PDF, image, UTF-8 text, and two-column CSV parsing with required human review.
- Per-field training-range validation. Invalid fields receive an inline error and red outline;
  prediction and history saving remain blocked until every value is valid.
- Highlighted predicted dataset label, mapped risk class, component explanation, readable decision
  path, leaf support, and confidence estimate.
- Reset control that returns every CBC input to zero.
- Collapsible sidebar containing Prediction, Test history, and Data quality pages.
- Local SQLite history with UTC time, model ID, source, diagnosis label, mapped risk class,
  confidence, support state, and all measurement values. History can be downloaded or cleared.
- Developer-only experiment artifacts for SOP 1 feature selection, SOP 2 node selection, and SOP 3
  branching and sensitivity. These analyses are intentionally absent from the end-user sidebar.

## Risk classes

Risk is assigned from the predicted dataset diagnosis; it is not calculated from validated clinical
thresholds.

- Class 1 - Low Risk: Healthy
- Class 2 - Moderate Risk: Normocytic hypochromic anemia, Normocytic normochromic anemia, Iron
  deficiency anemia, Other microcytic anemia, and Macrocytic anemia
- Class 3 - High Risk: Leukemia, Thrombocytopenia, and Leukemia with thrombocytopenia

## Confidence meaning

Confidence is the Laplace-smoothed share of training labels at the reached leaf:

`(matching leaf labels + 1) / (leaf samples + number of classes)`

The estimate is reduced by 25% when any used input is outside the feature range observed during
training. Levels are High at 80% or above, Moderate at 60% to below 80%, and Low below 60%.
This is an explanation of model agreement, not a calibrated clinical probability.

On the 303-row held-out test set, the generated reliability report recorded 274 High-confidence
predictions at 99.64% accuracy, 16 Moderate predictions at 100%, and 13 Low predictions at
84.62%. Small groups make these descriptive results uncertain.

## Main structure

```text
app.py                         Streamlit interface
src/hematology/
  constants.py                Schema and stable paths
  data.py                     Loading, validation, audit, and splitting
  models.py                   Numerical ID3, nodes, confidence, explanations
  analysis.py                 SOP 1, SOP 2, and SOP 3 analyses
  evaluation.py               Metrics and cross-validation
  examples.py                 Five training-only examples per class
  history.py                  Local SQLite prediction history
  interpretation.py           Risk mapping and component explanations
  report_parser.py            PDF/OCR parsing and extraction safeguards
  service.py                  Application orchestration and selected model loading
scripts/
  audit_data.py               Non-destructive dataset audit
  run_experiments.py          Model selection and all SOP exports
tests/                        Unit and integration regression tests
configs/                      Baseline and experiment configuration records
artifacts/latest/             Measured experiment outputs
docs/                         Architecture, SOP methods, rules, and user guide
```

Earlier demonstration and screening scripts remain in the repository for research traceability.
The Streamlit application uses only the shared implementation under `src/hematology/`.

## Version 1.2 corrections

- Stable keyed field containers produce red outlines and errors above and beside field names.
- Nonnumeric text such as `sadaa` is explained and rejected, along with missing, non-finite and
  out-of-range values. All 14 fields are checked before model inference or a history write.
- Diagnosis and requested risk class are both highlighted. Decision steps are readable text.
- A result is invalidated on input changes; ordinary rerenders do not save it again.
- Partial reports clear missing fields instead of carrying over another record's measurements.
- All 14 canonical CSV names are supported, including LYMp/LYMn and NEUTp/NEUTn.
- A reference-only interval cannot be used as the measured result.
- Component comparisons include the actual value and Healthy training median. These are not
  clinical thresholds, deficiency diagnoses or treatment targets.
- Model evaluation and SOP analysis are excluded from the sidebar; developer scripts remain.
- `samples/cbc_training_example.csv` supplies an existing Healthy training example for upload.
  `samples/cbc_upload_template.csv` is a blank fourteen-field template.
- Streamlit is pinned to the version used for UI validation. Optional `ID3_HISTORY_PATH` selects
  a different SQLite path. The local prototype has no user accounts or access-control layer.
