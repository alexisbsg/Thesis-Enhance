from io import BytesIO

from hematology.report_parser import extract_cbc_values, read_uploaded_report


def test_absolute_lymphocyte_does_not_fill_percentage():
    values = extract_cbc_values("ABSOLUTE LYMPHOCYTE 2.1")
    assert values["LYMn"].value == 2.1
    assert "LYMp" not in values


def test_result_is_preferred_over_reference_range():
    values = extract_cbc_values("HGB Reference 12.0-16.0 Result 13.5")
    assert values["HGB"].value == 13.5


def test_unicode_unit_is_not_read_as_result():
    values = extract_cbc_values("WBC 10⁹/L 6.5")
    assert values["WBC"].value == 6.5


def test_utf8_text_report_can_be_read():
    report = BytesIO(b"WBC,7.2\nHGB,13.5\n")
    text = read_uploaded_report(report, "cbc.csv")
    extracted = extract_cbc_values(text)
    assert extracted["WBC"].value == 7.2
    assert extracted["HGB"].value == 13.5


def test_all_canonical_column_names_round_trip():
    from hematology.constants import FEATURES

    text = "test,result\n" + "\n".join(
        f"{name},{index + 1}.5" for index, name in enumerate(FEATURES)
    )
    parsed = extract_cbc_values(text)
    assert set(parsed) == set(FEATURES)
    assert all(parsed[name].value == index + 1.5 for index, name in enumerate(FEATURES))


def test_reference_only_line_does_not_become_a_measurement():
    assert extract_cbc_values("HGB Reference 12.0-16.0") == {}
    assert extract_cbc_values("HGB 12.0-16.0") == {}
