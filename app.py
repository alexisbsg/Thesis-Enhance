"""Version 1.2 interface. Run this file with python -m streamlit run app.py."""

from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st

from hematology.constants import DEFAULT_HISTORY_PATH, FEATURES, TARGET_COLUMN
from hematology.data import DataValidationError, record_validation_errors
from hematology.history import PredictionHistory
from hematology.interpretation import component_comparison, explain_components, risk_for_label
from hematology.report_parser import extract_cbc_values, read_uploaded_report
from hematology.service import SavedPrediction, build_system, predict_and_save

APP_VERSION = "1.2.0"
st.set_page_config(
    page_title="CBC ID3 Research System",
    page_icon="🩸",
    layout="wide",
    initial_sidebar_state="collapsed",
)
st.markdown(
    """<style>
    .prediction-card {border:3px solid #2563eb;border-radius:14px;padding:1.2rem;
        margin:1rem 0;background:#eff6ff;color:#172554;}
    .result-value {font-size:2rem;font-weight:800;}
    .risk-value {font-size:1.25rem;font-weight:700;}
    .field-error {color:#dc2626;font-weight:650;margin-bottom:0.25rem;}
    [data-testid="stExpandSidebarButton"] span,
    [data-testid="stSidebarCollapseButton"] button span {display:none;}
    [data-testid="stExpandSidebarButton"]::after,
    [data-testid="stSidebarCollapseButton"] button::after {content:"☰";font-size:1.6rem;}
    </style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_system():
    return build_system()


@st.cache_resource
def load_history() -> PredictionHistory:
    return PredictionHistory(DEFAULT_HISTORY_PATH)


try:
    system = load_system()
    history = load_history()
except Exception as error:
    st.error(f"System initialization failed: {error}")
    st.stop()


def input_key(feature: str) -> str:
    return f"cbc_input_{feature}"


def clear_result() -> None:
    """Invalidate an old result whenever its input or confirmation changes."""
    st.session_state.pop("saved_prediction", None)


def input_changed() -> None:
    clear_result()
    st.session_state["units_confirmed"] = False
    st.session_state["input_source"] = "manual"
    st.session_state["draft"] = {
        feature: st.session_state.get(input_key(feature), "0") for feature in FEATURES
    }


def set_record(record: dict[str, float], source: str = "manual") -> None:
    """Replace the entire record, including missing report fields; never mix patients."""
    clear_result()
    st.session_state["draft"] = {
        feature: str(record[feature]) if feature in record else "" for feature in FEATURES
    }
    for feature, value in st.session_state["draft"].items():
        st.session_state[input_key(feature)] = value
    st.session_state["units_confirmed"] = False
    st.session_state["attempted"] = False
    st.session_state["input_source"] = source
    st.session_state.pop("extraction_review", None)
    st.session_state.pop("submission_error", None)


def reset_inputs() -> None:
    set_record(dict.fromkeys(FEATURES, 0))


def render_result(saved: SavedPrediction) -> None:
    """Render the saved result without inserting another history row on a rerun."""
    explanation = saved.explanation
    risk_number, risk_name = risk_for_label(explanation.prediction)
    st.markdown(
        '<section class="prediction-card" role="status">'
        "<div>PREDICTED DATASET DIAGNOSIS</div>"
        f'<div class="result-value">{escape(explanation.prediction)}</div>'
        f'<div class="risk-value">{escape(risk_number)} — {escape(risk_name)}</div>'
        "</section>",
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    columns[0].metric("Confidence estimate", f"{explanation.confidence:.1%}")
    columns[1].metric("Confidence level", explanation.confidence_level)
    columns[2].metric("Training rows in leaf", explanation.leaf_samples)
    st.caption(
        "Confidence describes smoothed training-label agreement in this leaf, "
        "not a calibrated medical probability."
    )
    st.subheader("What the result means")
    interpretation = explain_components(explanation.prediction, saved.record, system.training_data)
    st.write(interpretation["description"])
    st.write(
        "The comparison below shows which relevant measurements are lower or higher than "
        "the Healthy training-group median, and the actual values behind that comparison."
    )
    st.dataframe(
        component_comparison(
            explanation.prediction,
            saved.record,
            system.training_data,
        ),
        hide_index=True,
        width="stretch",
    )
    st.info(
        "A lower dataset comparison does not prove a deficiency or mean a component must be "
        "increased. Use the laboratory's reference ranges and a clinician's assessment to "
        "decide what needs investigation or treatment."
    )
    with st.expander("Decision path", expanded=True):
        # SOP 2 presentation: show the decisions recorded by explain_one as readable steps.
        # Internal dictionaries remain useful for research exports, never as the result UI.
        for index, decision in enumerate(explanation.decision_path, 1):
            st.markdown(f"**Step {index}:** {decision}")
        st.markdown(
            f"**Final leaf:** {explanation.prediction} "
            f"({explanation.leaf_samples} training records)."
        )
        total = sum(explanation.class_counts.values())
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Dataset class": label,
                        "Training rows": count,
                        "Leaf share": f"{count / total:.1%}",
                    }
                    for label, count in sorted(explanation.class_counts.items())
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption(f"Saved as test history record {saved.history_id}.")


with st.sidebar:
    st.title("☰ CBC menu")
    page = st.radio("Navigation", ["Prediction", "Test history", "Data quality"])
    st.caption(f"App version {APP_VERSION}")
    st.caption(f"Model: {system.model_id}")
    st.warning("Research prototype. Not validated for clinical diagnosis.")

st.caption(f"CBC Research • Version {APP_VERSION} • Use ☰ at the top left to open the menu")
if page == "Prediction":
    st.title("CBC classification and explanation")
    st.caption(
        "Risk class is assigned from the predicted dataset diagnosis. "
        "It is not calculated from validated clinical thresholds."
    )
    with st.expander("Risk class mapping"):
        st.markdown(
            "**Class 1 — Low Risk:** Healthy\n\n"
            "**Class 2 — Moderate Risk:** the five anemia labels in this dataset\n\n"
            "**Class 3 — High Risk:** Leukemia, Thrombocytopenia, "
            "and Leukemia with thrombocytopenia"
        )
    left, right = st.columns(2)
    with left:
        st.subheader("Autofill an existing example")
        label = st.selectbox("Dataset class", sorted(system.examples))
        number = st.selectbox("Example", range(1, len(system.examples[label]) + 1))
        st.button(
            "Load example values",
            on_click=set_record,
            args=(system.examples[label][number - 1], f"example:{label}:{number}"),
        )
    with right:
        st.subheader("Read a CBC report")
        uploaded = st.file_uploader(
            "PDF, image, UTF-8 TXT, or two-column CSV",
            type=["pdf", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "txt", "csv"],
        )
        if uploaded and st.button("Extract measurements"):
            try:
                uploaded.seek(0)
                extracted = extract_cbc_values(read_uploaded_report(uploaded, uploaded.name))
                if not extracted:
                    st.error("No supported measurements found. Current inputs were not replaced.")
                else:
                    set_record(
                        {name: item.value for name, item in extracted.items()}, "report_upload"
                    )
                    st.session_state["extraction_review"] = [
                        {"Feature": name, "Value": item.value, "Source line": item.source_line}
                        for name, item in extracted.items()
                    ]
                    st.rerun()
            except Exception as error:
                st.error(f"Could not extract report: {error}")
        with st.expander("Supported report format"):
            st.write(
                "Use one test and measured result per line, for example HGB,13.5. "
                "For differential counts use LYMPHOCYTE % and ABSOLUTE LYMPHOCYTE "
                "separately. Canonical names such as LYMp and LYMn also work. "
                "PDF text must be selectable (maximum 10 pages). Images need Tesseract OCR. "
                "Units are not converted automatically; review every value."
            )
            template = "test,result\n" + "\n".join(f"{name}," for name in FEATURES)
            st.download_button(
                "Download blank CBC CSV template", template, "cbc_upload_template.csv", "text/csv"
            )
    if st.session_state.get("extraction_review"):
        st.warning("Check extracted measurements. Missing fields are blank and must be completed.")
        st.dataframe(
            pd.DataFrame(st.session_state["extraction_review"]), hide_index=True, width="stretch"
        )
    for feature in FEATURES:
        st.session_state.setdefault(
            input_key(feature), st.session_state.get("draft", {}).get(feature, "0")
        )
    values = {feature: st.session_state[input_key(feature)] for feature in FEATURES}
    errors = record_validation_errors(values, system.input_ranges)
    visible_errors = errors if st.session_state.get("attempted") else {}
    # Use stable container keys for error styling; labels and entered text never become selectors.
    styles = "\n".join(
        f".st-key-field_{feature} input {{border:2px solid #dc2626 !important;}}"
        for feature in visible_errors
    )
    st.markdown(f"<style>{styles}</style>", unsafe_allow_html=True)
    st.subheader("CBC measurements")
    st.caption("Accepted ranges describe this model's training data, not clinical normal ranges.")
    if visible_errors:
        st.error("Prediction blocked. Correct the highlighted fields. No history record was saved.")
    columns = st.columns(3)
    for index, feature in enumerate(FEATURES):
        with columns[index % 3], st.container(key=f"field_{feature}"):
            low, high = system.input_ranges[feature]
            field_label = (
                f"<strong>{feature}</strong> "
                f'<span style="color: #808080;">({low:g} to {high:g})</span>'
            )
            message = visible_errors.get(feature)

            if message:
                st.markdown(
                    f'<div class="field-error">{escape(message)}</div>',
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f'{field_label} '
                    f'<span class="field-error">({escape(message)})</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(field_label, unsafe_allow_html=True)

            st.text_input(
                feature,
                key=input_key(feature),
                label_visibility="collapsed",
                on_change=input_changed,
            )
    confirmed = st.checkbox(
        "I reviewed all values and confirmed the dataset's expected units.",
        key="units_confirmed",
        on_change=clear_result,
    )
    predict_column, reset_column = st.columns(2)
    submitted = predict_column.button("Predict and save to history", type="primary")
    reset_column.button("Reset inputs to 0", on_click=reset_inputs)
    if submitted:
        st.session_state["attempted"] = True
        clear_result()
        try:
            saved = predict_and_save(
                system,
                history,
                values,
                units_confirmed=confirmed,
                input_source=st.session_state.get("input_source", "manual"),
            )
            st.session_state["saved_prediction"] = saved
            st.session_state.pop("submission_error", None)
        except DataValidationError as error:
            st.session_state["submission_error"] = (
                "Correct the highlighted fields before predicting." if errors else str(error)
            )
        st.rerun()
    if st.session_state.get("submission_error") and not visible_errors:
        st.error(st.session_state["submission_error"])
    if saved := st.session_state.get("saved_prediction"):
        render_result(saved)

elif page == "Test history":
    st.title("Prediction test history")
    records = history.list()
    if records.empty:
        st.info("No predictions have been saved yet.")
    else:
        display = records.copy()
        display["confidence"] = display["confidence"].map(lambda value: f"{value:.1%}")
        st.dataframe(display, hide_index=True, width="stretch")
        st.download_button(
            "Download history CSV",
            records.to_csv(index=False).encode(),
            "prediction_history.csv",
            "text/csv",
        )
        if st.button("Clear local history"):
            history.clear()
            clear_result()
            st.rerun()
    st.caption("History contains local test inputs and results, without patient names.")
else:
    st.title("Data quality")
    st.write(
        "These checks describe the model's dataset: missing or non-finite values, duplicate "
        "rows, identical measurements with conflicting labels, class counts and a file "
        "fingerprint. They do not assess whether an individual CBC is clinically normal."
    )
    st.dataframe(
        pd.DataFrame(
            [
                {"Check": key.replace("_", " ").title(), "Result": str(value)}
                for key, value in system.audit.to_dict().items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    st.dataframe(
        system.data[TARGET_COLUMN].value_counts().rename_axis("Class").reset_index(name="Rows"),
        hide_index=True,
        width="stretch",
    )
