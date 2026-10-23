"""Dataset-label risk mapping and cautious, human-readable result explanations."""

from __future__ import annotations

import pandas as pd

from .constants import ANEMIA_LABELS, HIGH_RISK_LABELS, TARGET_COLUMN

RISK_CLASSES: dict[str, tuple[str, str]] = {
    "Healthy": ("Class 1", "Low Risk"),
    **{label: ("Class 2", "Moderate Risk") for label in ANEMIA_LABELS},
    **{label: ("Class 3", "High Risk") for label in HIGH_RISK_LABELS},
}

COMPONENT_FOCUS: dict[str, tuple[str, ...]] = {
    "Healthy": ("WBC", "RBC", "HGB", "HCT", "PLT"),
    "Normocytic hypochromic anemia": ("RBC", "HGB", "HCT", "MCV", "MCHC"),
    "Normocytic normochromic anemia": ("RBC", "HGB", "HCT", "MCV", "MCHC"),
    "Iron deficiency anemia": ("RBC", "HGB", "HCT", "MCV", "MCH", "MCHC"),
    "Other microcytic anemia": ("HGB", "HCT", "MCV", "MCH", "MCHC"),
    "Macrocytic anemia": ("RBC", "HGB", "HCT", "MCV"),
    "Leukemia": ("WBC", "LYMp", "NEUTp", "LYMn", "NEUTn"),
    "Thrombocytopenia": ("PLT", "PDW", "PCT"),
    "Leukemia with thrombocytopenia": (
        "WBC",
        "LYMp",
        "NEUTp",
        "LYMn",
        "NEUTn",
        "PLT",
        "PCT",
    ),
}

LABEL_DESCRIPTIONS: dict[str, str] = {
    "Healthy": "The entered pattern is most similar to the Healthy records in this dataset.",
    "Normocytic hypochromic anemia": (
        "The dataset label describes an anemia-like pattern with cell size closer to the dataset's "
        "normocytic group and lower hemoglobin concentration indicators."
    ),
    "Normocytic normochromic anemia": (
        "The dataset label describes an anemia-like pattern whose cell size and hemoglobin "
        "concentration indicators are closer to the dataset's normocytic and normochromic groups."
    ),
    "Iron deficiency anemia": (
        "The dataset label is associated with an anemia-like pattern involving hemoglobin and red "
        "cell indices such as MCV, MCH, and MCHC."
    ),
    "Other microcytic anemia": (
        "The dataset label describes an anemia-like pattern with smaller red-cell indices, "
        "particularly MCV and MCH."
    ),
    "Macrocytic anemia": (
        "The dataset label describes an anemia-like pattern with a higher red-cell size indicator, "
        "particularly MCV."
    ),
    "Leukemia": (
        "The dataset label is associated with patterns in white blood cell and differential counts."
    ),
    "Thrombocytopenia": (
        "The dataset label is associated mainly with platelet-related measurements such as "
        "PLT and PCT."
    ),
    "Leukemia with thrombocytopenia": (
        "The dataset label combines white-cell or differential patterns with "
        "platelet-related patterns."
    ),
}


def risk_for_label(label: str) -> tuple[str, str]:
    """Return the user-approved risk class assigned from the predicted dataset label."""
    try:
        return RISK_CLASSES[label]
    except KeyError as error:
        raise ValueError(f"No risk mapping is defined for dataset label: {label}") from error


def explain_components(
    label: str,
    record: dict[str, float],
    training_data: pd.DataFrame,
) -> dict[str, str | list[str]]:
    """Compare relevant inputs with Healthy medians without implying clinical thresholds."""
    healthy = training_data.loc[training_data[TARGET_COLUMN] == "Healthy"]
    if healthy.empty:
        return {
            "description": LABEL_DESCRIPTIONS[label],
            "lower": [],
            "higher": [],
        }
    medians = healthy.median(numeric_only=True)
    lower: list[str] = []
    higher: list[str] = []
    for feature in COMPONENT_FOCUS[label]:
        median = float(medians[feature])
        value = float(record[feature])
        if value < median * 0.90:
            lower.append(feature)
        elif value > median * 1.10:
            higher.append(feature)
    return {
        "description": LABEL_DESCRIPTIONS[label],
        "lower": lower,
        "higher": higher,
    }


def component_comparison(
    label: str,
    record: dict[str, float],
    training_data: pd.DataFrame,
) -> pd.DataFrame:
    """Explain component direction with actual values, without prescribing treatment.

    Healthy medians come only from training rows. They are descriptive model
    context, not clinical reference intervals or targets for the tested person.
    """
    healthy = training_data.loc[training_data[TARGET_COLUMN] == "Healthy"]
    rows = []
    for feature in COMPONENT_FOCUS[label]:
        value = float(record[feature])
        median = float(healthy[feature].median()) if not healthy.empty else None
        if median is None:
            comparison = "No Healthy training reference"
        elif value < median:
            comparison = "Below dataset median"
        elif value > median:
            comparison = "Above dataset median"
        else:
            comparison = "At dataset median"
        rows.append(
            {
                "Component": feature,
                "Your value": value,
                "Healthy training median": median,
                "Comparison": comparison,
            }
        )
    return pd.DataFrame(rows)
