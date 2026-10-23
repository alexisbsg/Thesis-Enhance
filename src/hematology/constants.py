"""Stable schema and research configuration."""

import os
from pathlib import Path

FEATURES: tuple[str, ...] = (
    "WBC",
    "LYMp",
    "NEUTp",
    "LYMn",
    "NEUTn",
    "RBC",
    "HGB",
    "HCT",
    "MCV",
    "MCH",
    "MCHC",
    "PLT",
    "PDW",
    "PCT",
)
TARGET_COLUMN = "Diagnosis"
DEFAULT_SEED = 42
TEST_SIZE = 0.25
DATASET_NAME = "diagnosed_cbc_screened.csv"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = PROJECT_ROOT / DATASET_NAME
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_HISTORY_PATH = Path(
    os.environ.get("ID3_HISTORY_PATH", PROJECT_ROOT / "runtime" / "prediction_history.sqlite3")
)

# The repository does not document authoritative units. Showing guessed units
# would make input errors more likely, so the UI uses this explicit placeholder.
FEATURE_UNITS = {feature: "dataset-native unit" for feature in FEATURES}

ANEMIA_LABELS: frozenset[str] = frozenset(
    {
        "Normocytic hypochromic anemia",
        "Normocytic normochromic anemia",
        "Iron deficiency anemia",
        "Other microcytic anemia",
        "Macrocytic anemia",
    }
)
HIGH_RISK_LABELS: frozenset[str] = frozenset(
    {"Leukemia", "Thrombocytopenia", "Leukemia with thrombocytopenia"}
)
