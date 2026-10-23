# ID3 Hematological Codebase and SOP Guide

Version 1.2.0 • 23 October 2026

This guide explains how to run the corrected application, how information moves through its code, and why the three SOP solutions use feature ranking, transparent node selection, and controlled recursive branching. It also explains the risk classes, validation, history, confidence, report uploads, and limits of the measured results.

The application keeps all nine dataset diagnoses and the requested three risk classes. Invalid measurements cannot reach prediction or history saving through the application service. Model evaluation and SOP analyses are developer tools and are absent from the sidebar. The sidebar contains Prediction, Test history, and Data quality.

## Start the corrected version

1. Stop an older Streamlit process with Ctrl+C in its terminal. Close its browser tab.
2. Extract the new ZIP into a fresh folder. Do not run the project while it is still compressed.
3. On Windows, double-click run_app.bat. It creates a local environment, installs this project, and runs this folder's app.py. Python 3.11 or newer and internet access for installation are required.
4. On macOS or Linux, run bash run_app.sh from the extracted folder.
5. Open the URL printed in the terminal. Confirm that the page says Version 1.2.0. The hamburger control is at the top left.

For a manual start, create and activate a virtual environment, run python -m pip install -e ., then python -m streamlit run app.py. Installing only requirements.txt is insufficient for the package imports unless you also install the project. The launchers do both through the editable installation.

## A short acceptance walkthrough

Choose Healthy and an example, click Load example values, check the unit confirmation, and predict. The highlighted panel should show a dataset diagnosis and mapped risk class. Expand Decision path to see sentences, not JSON. Open Test history to find the saved record.

Replace HGB with sadaa, or a value above its displayed maximum, and submit again. The field must receive a red outline and error text above and beside its name. No new result or history row should appear. Reset inputs to 0 returns all 14 fields to zero, clears the result and error display, and removes unit confirmation. Zero is a reset state, not a valid complete CBC.

Your previous local history is not bundled into the ZIP. To preserve it when moving folders, stop the old app and copy runtime/prediction_history.sqlite3 to the same relative location in the new folder. Existing history tables receive the risk column automatically.

---PAGE---
# Codebase responsibilities and flow

The UI delegates work to a service, which validates inputs before calling the model and database. The mathematical model is independent of Streamlit. This separation makes the code easier to test and prevents a display change from silently changing the learning algorithm.

| File under the project root | Main responsibility |
| --- | --- |
| app.py | Navigation, fields, inline errors, result presentation and reset |
| src/hematology/constants.py | Fourteen feature names, paths and diagnosis groups |
| src/hematology/data.py | Dataset loading, validation, audit and stratified split |
| src/hematology/models.py | Entropy, candidate scores, tree growth and prediction |
| src/hematology/analysis.py | Feature ranking, node reports, stability and sensitivity |
| src/hematology/evaluation.py | Per-class metrics and fold-local cross-validation |
| src/hematology/service.py | Build the model and validate, predict and save a record |
| src/hematology/interpretation.py | Risk mapping, label descriptions and component comparisons |
| src/hematology/report_parser.py | PDF or image text extraction and measurement parsing |
| src/hematology/examples.py | Five existing training records per diagnosis |
| src/hematology/history.py | SQLite storage, migration, listing and deletion |
| scripts/run_experiments.py | Developer experiments and reproducible reports |
| tests/ | Mathematical, parsing, service and UI regression checks |

## Startup

build_system loads diagnosed_cbc_screened.csv, checks the schema and values, and makes the fixed stratified historical training/test split. It reads the selected configuration and feature list from artifacts/latest/selected_model.json when available, then fits NumericalID3Classifier on the training partition. This implementation trains at startup and caches the fitted object; the JSON stores configuration and evaluation, not a serialized fitted tree.

The training partition supplies input minima and maxima, example records and Healthy comparison medians. The test partition does not supply these runtime settings. With the packaged artifacts the model uses ten features, while the input form and history retain all fourteen columns.

## One prediction

Manual entry, a preset or reviewed extraction supplies a complete record. predict_and_save calls validate_record, verifies unit confirmation, calls explain_one, maps the returned label to a risk class, and inserts the result into SQLite. It returns a SavedPrediction containing the cleaned record, explanation and history ID.

render_result displays this object. Rendering it again does not write a second history row. Changing a measurement, loading another example, importing another report or resetting invalidates the old displayed result.

---PAGE---
# SOP 1 Feature relevance and selection

The question is which CBC measurements contribute to classification, and whether fewer measurements can preserve performance. A feature should not be retained or removed merely because its name seems medically important, or because it happened to be the root of one tree.

## Where the logic is implemented

In analysis.py, feature_ranking calls best_split separately for each feature. The model tests midpoint thresholds between adjacent distinct values and records the best information gain. select_top_features takes the first k features from this ranking. feature_usage counts how frequently each feature appears across the fitted tree, which provides a different view from root-level ranking.

In evaluation.py, cross_validate_configurations receives a feature_selector callback. It calls this callback on fold_train, not on all development records. The selected features then train that fold's model and are evaluated on fold_validation.

```python
selected_features = feature_selector(fold_train)
model.fit(fold_train, selected_features, TARGET_COLUMN)
predicted = model.predict(fold_validation.loc[:, selected_features])
```

The snippet summarizes the order of operations. Comments beginning SOP 1 identify the corresponding implementation and explain the separation between ranking and evaluation.

## Why feature selection happens inside each fold

If feature ranking used validation labels before evaluation, the validation set would influence the model indirectly. Recomputing the ranking inside every training fold avoids this specific leakage path. The separate historical test partition is evaluated after a candidate has been chosen.

scripts/run_experiments.py compares feature counts of 8, 10, 12 and 14 with four tree configurations using three stratified folds and one repeat. It selects the largest mean macro-F1. On a tie it prefers fewer features, then fewer mean leaves. Macro-F1 gives each diagnosis equal weight, so a common class cannot dominate the score as easily as it can dominate accuracy.

## What the selected model uses

The supplied selection uses HGB, MCV, MCHC, MCH, RBC, HCT, PLT, NEUTp, PCT and WBC. LYMp, LYMn, NEUTn and PDW remain in the form and history but do not make decisions in this selected classifier. All fourteen are still required by the application schema and validation rule.

The ranking and usage exports are sop1_feature_ranking.csv and sop1_feature_usage.csv. A weak root score does not prove a feature is universally useless: it may help deeper in a branch or interact with another feature. This is why ranking, whole-tree usage and measured subset evaluation are reported separately.

---PAGE---
# SOP 2 Entropy and node selection

SOP 2 asks how a feature and threshold become a decision node. The implemented solution makes that choice reproducible and inspectable. It does not assume that entropy is mathematically broken or that overlapping class values necessarily make the root unstable.

## Entropy and information gain

entropy in models.py counts the labels and calculates Shannon entropy in bits: minus the sum of p multiplied by log2(p). A node containing only one label has entropy zero. A mixed node has greater uncertainty. An empty collection returns zero so the helper has a defined edge case.

score_candidates sorts each numerical feature's distinct values and uses the midpoints between neighboring values as thresholds. For every eligible threshold, records at or below it go left and larger values go right. A candidate's score is:

Information gain = parent entropy − weighted child entropy

Weighted child entropy = (left count / total) × left entropy + (right count / total) × right entropy

For ten records containing six A and four B labels, parent entropy is about 0.971. If the left branch contains five A and the right contains one A and four B, the weighted child entropy is about 0.361. Information gain is therefore about 0.610. Weighting matters because a large mixed branch must not be treated as if it were as small as a tiny pure branch.

## Selecting and explaining a node

best_split retains the candidate with the largest information gain. Its strict greater-than comparison keeps the first candidate on an exact tie. Feature order and ascending thresholds give a deterministic tie rule. When a node becomes terminal, its prediction is the majority label; the existing pandas mode ordering resolves tied majorities.

TreeNode stores entropy, sample count, class counts, prediction, depth, selected feature, threshold, gain and stopping reason. node_selection_report exports these values to sop2_selected_nodes.csv. Candidate scores are calculated by score_candidates; the current export contains the chosen nodes, not a permanent log of every rejected candidate.

explain_one follows the fitted branches for a new record and builds readable decisions such as HGB (12.5) <= 12.95. app.py renders these as numbered steps and a final leaf. Class composition appears in a table rather than raw JSON.

## What stability means

root_stability repeats baseline root selection using seeds 42, 7, 21, 100 and 2026. The supplied descriptive report selected HGB at 12.95 in all five runs. This checks changing training membership; deterministic tie-breaking checks repeated choices for the same data. They answer different questions. The stability report is descriptive and is not used to tune the final model.

---PAGE---
# SOP 3 Recursive branching and leaf size

A decision tree repeatedly splits records until it reaches endpoints called leaves. Very small leaves can describe only a few training examples. SOP 3 introduces controls that limit this behavior, then measures the performance and complexity tradeoff instead of assuming a smaller tree is always better.

## The controls in models.py

| Setting | What it controls | Packaged selection |
| --- | --- | --- |
| max_depth | Maximum root-to-leaf edge depth; root depth is zero | 10 |
| min_samples_split | Minimum parent records before attempting a split | 6 |
| min_samples_leaf | Minimum records in each proposed child | 3 |
| min_information_gain | Gain must exceed this value to split | 0.0 |

_build first creates a node with its majority prediction. It then checks pure labels, depth and parent size. It searches valid candidates, stops if there is no candidate or insufficient gain, and otherwise creates left and right children recursively. Every stop stores a reason, making the termination explainable.

score_candidates enforces min_samples_leaf before accepting a candidate score. With min_samples_leaf=3, a six-record parent can split 3/3, but cannot split 5/1 or 4/2. Rejecting one threshold does not stop the whole search; another threshold or feature may still work.

If an initial training dataset has fewer records than the configured leaf minimum, the unsplit root can itself be smaller. The setting constrains children created by splitting; it does not manufacture missing training records. In the packaged fit, the actual depth is 5 despite the limit of 10.

## Why this is pre-pruning

The controls prevent some branches from being created. That is pre-pruning. This project does not implement a separate algorithm that grows a full tree and later removes branches. The source comments distinguish the checks so you can explain the actual algorithm accurately in your thesis.

scripts/run_experiments.py compares four configurations on the same training folds and combines this with the feature-subset experiment. The selected leaf minimum is 3 and split minimum is 6. The fitted tree still has 14 internal nodes and 15 leaves; do not claim that these settings made this particular final tree smaller than the original tree when the measured counts are equal.

## Sensitivity testing

sensitivity_analysis applies seeded positive or negative perturbations to measurements. Each perturbation is alpha times a feature's training standard deviation. The test reports how often the fixed model changes its prediction. The supplied mean change rates are about 0.89%, 1.06% and 6.73% for alpha 0.005, 0.01 and 0.02 respectively.

These are artificial stress tests, not validated models of laboratory measurement error. Larger leaf sizes can also hurt rare classes, so complexity and per-class metrics must be considered together.

---PAGE---
# Input validation and history protection

The form accepts text so an entry such as sadaa can receive a specific numeric error instead of being silently transformed by a browser number widget. The record is converted to numeric values only after validation. This does not change the model's numeric inputs.

## Field errors and accepted ranges

record_validation_errors in data.py returns a dictionary of feature names and messages. It checks missing values, numeric conversion, finite values and inclusive training bounds. validate_record raises DataValidationError when this dictionary is nonempty and otherwise returns the fourteen features in canonical order.

An accepted training bound is not a clinical normal interval. A value outside the training envelope may be a genuine measurement; the app refuses to classify it because this build is configured to block inputs outside model support. The user should not change a real measurement simply to pass validation.

app.py displays validation messages after a submission attempt. Each field has a stable container key such as field_HGB. The CSS selector targets that container's input, so error styling no longer depends on the changing error message or the text entered by the user. The message is printed above the field and in parentheses beside its name.

## The service is the enforcement point

```python
clean = validate_record(record, system.input_ranges)
if not units_confirmed:
    raise DataValidationError(...)
explanation = system.model.explain_one(clean)
# Map risk and insert history only after both gates succeed.
```

The service ordering is deliberate. Tests replace explain_one with a spy and confirm it is never called for invalid inputs or unconfirmed units. Those tests also verify that the history table remains empty. Validation covers all fourteen fields even when the selected model uses only ten.

## Preventing stale or mixed records

set_record replaces every field. If an uploaded report contains only HGB, the other thirteen inputs become blank rather than retaining values from an earlier example or person. Imported values retain their source lines for review. No prediction proceeds until all fields are complete and valid.

input_changed removes the displayed result and clears unit confirmation. reset_inputs sets all fourteen values to zero and clears extraction evidence, submission errors, the displayed result and confirmation. It does not delete previous history. Saved results stay visible across ordinary rerenders without saving again; clicking Predict again intentionally creates another test record.

## What SQLite stores

history.py stores the model identifier, UTC timestamp, source, predicted diagnosis, mapped risk, confidence, confidence level, leaf support and all input values. It uses parameterized SQL. Existing tables are migrated to add risk_class and backfill known labels. Test history displays and exports the latest 500 records; older rows remain in the database until history is cleared. Clear local history deletes all rows.

---PAGE---
# Understanding the prediction result

The diagnosis, risk class, confidence and component comparisons describe different aspects of the result. Keeping them separate prevents a model support estimate from being mistaken for medical severity.

## Requested risk mapping

| Predicted dataset diagnosis | Displayed risk class |
| --- | --- |
| Healthy | Class 1 — Low Risk |
| Normocytic hypochromic anemia | Class 2 — Moderate Risk |
| Normocytic normochromic anemia | Class 2 — Moderate Risk |
| Iron deficiency anemia | Class 2 — Moderate Risk |
| Other microcytic anemia | Class 2 — Moderate Risk |
| Macrocytic anemia | Class 2 — Moderate Risk |
| Leukemia | Class 3 — High Risk |
| Thrombocytopenia | Class 3 — High Risk |
| Leukemia with thrombocytopenia | Class 3 — High Risk |

risk_for_label in interpretation.py performs this exact lookup. The original diagnosis is never replaced. Unknown labels raise an error instead of receiving an invented risk. The class is assigned from the diagnosis label and is not calculated from validated clinical thresholds.

## Confidence and leaf support

explain_one calculates (matching labels + 1) / (leaf records + number of classes). This Laplace smoothing avoids displaying 100% for every pure leaf, especially a tiny one. With nine classes and three matching records out of three, the estimate is 4/12, or 33.3%, despite the leaf being pure.

High means at least 80%; Moderate means at least 60% but below 80%; Low means below 60%. These are display bands, not calibrated probabilities or medical urgency levels. The model helper also reduces confidence by 25% for used features outside training support, but the application service blocks out-of-range records before inference, so that fallback is not normally reached through the UI.

## Explaining lower and higher components

LABEL_DESCRIPTIONS supplies a description of each dataset label. COMPONENT_FOCUS identifies relevant features. component_comparison shows the entered value, the Healthy training-group median and whether the value is below, at or above that median. For example, an anemia-related result focuses on HGB and red-cell indices, while a platelet-related result includes PLT and PCT.

This helps the user see which components deserve review, but it cannot establish what the person must increase. A median is not a lower clinical limit. Treatment depends on the cause and clinical context; the interface therefore provides comparisons rather than instructions to increase blood components. MedlinePlus explains that CBC interpretation may require symptoms, history and other tests; see the references at the end of this guide.

---PAGE---
# Uploading a readable CBC report

The most predictable input is UTF-8 text or a two-column CSV with one measurement per line. A CSV header such as test,result is optional. Use a decimal point. Upload one record at a time, not a spreadsheet containing multiple patients.

| Canonical test name | Meaning used in the project |
| --- | --- |
| WBC | White blood cell count |
| LYMp | Lymphocyte percentage |
| NEUTp | Neutrophil percentage |
| LYMn | Absolute lymphocyte count |
| NEUTn | Absolute neutrophil count |
| RBC | Red blood cell count |
| HGB | Hemoglobin |
| HCT | Hematocrit |
| MCV | Mean corpuscular volume |
| MCH | Mean corpuscular hemoglobin |
| MCHC | Mean corpuscular hemoglobin concentration |
| PLT | Platelet count |
| PDW | Platelet distribution width |
| PCT | Plateletcrit in this dataset |

For example, HGB,13.5 and LYMp,32.5 are two separate lines. The downloadable blank template contains all fourteen canonical names. samples/cbc_training_example.csv contains an existing Healthy training example for an end-to-end demonstration; it is not a clinical reference profile. Longer aliases such as LYMPHOCYTE % and ABSOLUTE LYMPHOCYTE also work.

## Formats and parsing logic

read_uploaded_report accepts selectable-text PDF with at most ten pages, PNG, JPG, JPEG, TIF, TIFF, BMP, UTF-8 TXT and CSV. Images require a separate Tesseract installation. A scanned PDF with no selectable text must be converted to a clear image before upload. The image limit is 25 megapixels.

extract_cbc_values identifies the longest matching test-name alias, isolates a result region, ignores unit exponents such as 10⁹/L, and avoids reading a reference-only interval as a result. Absolute counts and percentages have different aliases. Canonical names are supported so the CSV template round-trips through the parser.

The parser is intentionally limited: it keeps the first recognized measurement for each feature, does not infer missing values, and does not reliably understand arbitrary multi-column layouts or convert units. PCT is context-sensitive outside this dataset, so verify that a report uses it for plateletcrit. Review source lines, correct OCR mistakes, complete missing fields, and confirm units before predicting. The repository does not establish authoritative clinical units, so real-report unit compatibility must be verified separately.

---PAGE---
# Data quality and measured reliability

The Data quality page evaluates the dataset, not whether a tested person's blood results are normal. audit_dataset counts rows, features, classes, missing cells, non-finite values, exact duplicate rows and identical feature groups with conflicting diagnosis labels. It also computes a SHA-256 fingerprint of the dataset file so results can be tied to specific bytes.

The current audit finds 1,212 rows, 14 numerical features and 9 classes. It reports zero missing cells, zero non-finite cells, zero duplicate rows and zero conflicting feature groups. These checks do not prove that labels are medically correct, that patient records are independent, or that the dataset is representative of another population.

## What the removed pages used to do

Model evaluation showed metrics such as accuracy, per-class precision and recall, and a confusion matrix. SOP analyses showed feature relevance, node selection, root stability, complexity and sensitivity. These are useful for developers and thesis evidence, so the scripts and artifacts remain. Neither page is included in the user navigation.

## Reproduced results for this revision

| Measurement | Historical split result |
| --- | --- |
| Training records | 909 |
| Test records | 303 |
| Correct predictions | 300 |
| Accuracy | 99.01% |
| Balanced accuracy | 98.83% |
| Macro-F1 | 96.14% |
| Weighted-F1 | 99.09% |
| Tree structure | 14 internal nodes, 15 leaves, depth 5 |

These values were recomputed using the selected configuration during the revision. The changes to validation, navigation, comments and display do not alter the fitted model's mathematical decisions. The original baseline regression test also remains.

The reported training-only selection score is approximately 98.38% mean macro-F1 across three folds. Because this score helped choose a candidate, it is not an independent final estimate. The historical test split has also been examined in earlier work, so it should not now be described as a fresh external test.

The smallest class has only eleven dataset records and three in this historical test split. A few errors can substantially change its precision or recall. Overall accuracy cannot establish reliability for every class. External records, verified labels, patient-level independence, confirmed units and appropriate clinical validation are still needed before any real-world diagnostic claim.

To reproduce the developer exports, run python scripts/run_experiments.py. This retrains the candidates and overwrites artifacts/latest. Use a copy of the project if you want to retain the current evidence for comparison.

---PAGE---
# Reading the code and verifying changes

Read constants.py and data.py first to understand the schema and entry checks. Then follow service.py into models.py. Read interpretation.py and history.py to see how a mathematical prediction becomes a displayed result and saved record. Finally read analysis.py, evaluation.py and scripts/run_experiments.py for the research workflow.

## Find the SOP comments

Search the codebase for SOP 1, SOP 2 or SOP 3. The comments explain why each operation is needed.

| SOP | Main functions to read | Reason for the design |
| --- | --- | --- |
| SOP 1 | feature_ranking, feature_usage, select_top_features, cross_validate_configurations | Compare relevance while keeping selection inside training folds |
| SOP 2 | entropy, score_candidates, best_split, node_selection_report, root_stability | Make mathematical scoring, tie behavior and observed stability inspectable |
| SOP 3 | _build, score_candidates, complexity, sensitivity_analysis | Prevent unsupported tiny branches and measure the cost of stronger controls |

## Tests and developer commands

Install developer dependencies with python -m pip install -e ".[dev]". Run python -m pytest, then python -m ruff check app.py src scripts tests. Run python scripts/audit_data.py for the non-destructive dataset audit.

The tests cover entropy boundaries, minimum leaf size, smoothed confidence, baseline predictions, all nine risk mappings, database migration, invalid inputs, unit confirmation and valid history storage. Parser checks cover absolute-count separation, result versus reference-range handling, Unicode units and all fourteen canonical headers.

The Streamlit test covers prediction, risk display, readable paths, rerendering without duplicate saves, invalid text, reset and navigation. Browser checks verify actual red borders, the hamburger control and partial uploads.

## Implementation boundaries to preserve

Keep new prediction entry points behind predict_and_save. Do not bypass validation by calling explain_one directly from a new screen. Do not train on saved test history automatically, because those rows have predictions rather than independently verified diagnosis labels. Do not use historical test performance to repeatedly tune features or tree controls.

Keep uploaded reports and history databases out of Git and distributed archives. The app stores no patient names, but numerical health measurements still deserve access controls. This local prototype has no multi-user authentication. ID3_HISTORY_PATH can set a separate local database path for tests or deployment.

## References

Project source and artifacts establish implementation details and results. These references explain interface behavior and CBC interpretation, not clinical validation of this model.

Streamlit, st.container API reference, https://docs.streamlit.io/develop/api-reference/layout/st.container, accessed 23 October 2026. Keyed containers provide stable CSS class names.

MedlinePlus, Complete Blood Count CBC, https://medlineplus.gov/lab-tests/complete-blood-count-cbc/, accessed 23 October 2026. CBC findings require interpretation in clinical context and may require further tests.
