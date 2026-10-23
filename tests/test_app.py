from unittest.mock import patch

from streamlit.testing.v1 import AppTest

from hematology.constants import FEATURES
from hematology.history import PredictionHistory


def button(app, label):
    return next(item for item in app.button if item.label == label)


def test_ui_errors_result_reset_and_navigation(tmp_path):
    # Isolated storage prevents the test from modifying a user's local history.
    import streamlit as st

    st.cache_resource.clear()
    with patch("hematology.constants.DEFAULT_HISTORY_PATH", tmp_path / "history.sqlite3"):
        app = AppTest.from_file("../app.py", default_timeout=60).run()
        assert not app.exception
        assert app.radio[0].options == ["Prediction", "Test history", "Data quality"]
        assert len(app.text_input) == 14
        button(app, "Predict and save to history").click().run()
        assert app.error
        assert any("st-key-field_HGB input" in m.value for m in app.markdown)
        history = PredictionHistory(tmp_path / "history.sqlite3")
        assert history.list().empty
        app.selectbox[0].select("Healthy").run()
        button(app, "Load example values").click().run()
        app.checkbox[0].check().run()
        button(app, "Predict and save to history").click().run()
        assert not app.exception
        assert len(history.list()) == 1
        assert not app.get("json")
        assert any('class="prediction-card"' in m.value for m in app.markdown)
        assert any("Class 1 — Low Risk" in m.value for m in app.markdown)
        assert any("**Step 1:**" in m.value for m in app.markdown)
        app.run()
        assert len(history.list()) == 1
        app.text_input(key="cbc_input_HGB").set_value("sadaa").run()
        assert not any('class="prediction-card"' in m.value for m in app.markdown)
        button(app, "Predict and save to history").click().run()
        assert len(history.list()) == 1
        assert any("Enter a numeric value" in m.value for m in app.markdown)
        button(app, "Reset inputs to 0").click().run()
        assert all(float(app.text_input(key=f"cbc_input_{name}").value) == 0 for name in FEATURES)
        assert not app.checkbox[0].value
        assert not app.error
        app.radio[0].set_value("Data quality").run()
        assert not app.exception
        app.radio[0].set_value("Test history").run()
        assert not app.exception
    st.cache_resource.clear()
