"""Conservative CBC report text extraction with mandatory human review."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import BinaryIO


@dataclass(frozen=True)
class ExtractedMeasurement:
    feature: str
    value: float
    source_line: str
    status: str = "needs_review"


ALIASES: dict[str, tuple[str, ...]] = {
    "WBC": ("WHITE BLOOD CELLS", "WBC"),
    "LYMp": ("LYMPHOCYTE PERCENT", "LYMPHOCYTES %", "LYMPHOCYTE %", "LYMP %"),
    "NEUTp": ("NEUTROPHIL PERCENT", "NEUTROPHILS %", "NEUTROPHIL %", "NEUT %"),
    "LYMn": (
        "ABSOLUTE LYMPHOCYTE",
        "LYMPHOCYTES #",
        "LYMPHOCYTE #",
        "LYMPHOCYTE COUNT",
        "LYM #",
        "LYM#",
    ),
    "NEUTn": (
        "ABSOLUTE NEUTROPHIL",
        "NEUTROPHILS #",
        "NEUTROPHIL #",
        "NEUTROPHIL COUNT",
        "NEUT #",
        "NEUT#",
    ),
    "RBC": ("RED BLOOD CELLS", "RBC"),
    "HGB": ("HEMOGLOBIN", "HGB", "HB"),
    "HCT": ("HEMATOCRIT", "HCT", "PCV"),
    "MCV": ("MCV",),
    "MCH": ("MCH",),
    "MCHC": ("MCHC",),
    "PLT": ("PLATELETS", "PLATELET", "PLT"),
    "PDW": ("PDW",),
    "PCT": ("PCT",),
}
# Canonical dataset headers must round-trip through the downloadable CSV template.
for _feature in ALIASES:
    ALIASES[_feature] = (*ALIASES[_feature], _feature.upper())

NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:\.\d+)?|\.\d+)")


def _best_label(line: str) -> tuple[str, re.Match[str]] | None:
    candidates: list[tuple[int, int, str, re.Match[str]]] = []
    for feature, aliases in ALIASES.items():
        for alias in aliases:
            match = re.search(rf"(?<![A-Z]){re.escape(alias)}(?![A-Z])", line)
            if match:
                candidates.append((len(alias), -match.start(), feature, match))
    if not candidates:
        return None
    _, _, feature, match = max(candidates, key=lambda candidate: candidate[:3])
    return feature, match


def _result_region(suffix: str) -> str:
    result_match = re.search(r"\b(?:RESULT|VALUE)\b\s*[:=]?\s*", suffix)
    return suffix[result_match.end() :] if result_match else suffix


def _first_measurement_number(text: str) -> float | None:
    # A reference-only line is not a measured result. Remove complete intervals,
    # including both endpoints, so the upper bound cannot become a result.
    text = re.split(r"\b(?:REFERENCE|REF|RANGE)\b", text, maxsplit=1)[0]
    text = re.sub(r"\d+(?:\.\d+)?\s*[-–]\s*\d+(?:\.\d+)?", "", text)
    for match in NUMBER_PATTERN.finditer(text):
        before = text[max(0, match.start() - 2) : match.start()]
        after = text[match.end() : match.end() + 3]
        if "^" in before or any(char in before for char in "⁰¹²³⁴⁵⁶⁷⁸⁹"):
            continue
        if after.lstrip().startswith(("^", "⁰", "¹", "²", "³", "⁴", "⁵", "⁶", "⁷", "⁸", "⁹")):
            continue
        if re.match(r"\s*[-–]\s*\d", after):
            continue
        return float(match.group())
    return None


def extract_cbc_values(text: str) -> dict[str, ExtractedMeasurement]:
    """Extract at most one value per feature and preserve its evidence line."""
    extracted: dict[str, ExtractedMeasurement] = {}
    for raw_line in text.splitlines():
        line = " ".join(raw_line.upper().split())
        labeled = _best_label(line)
        if labeled is None:
            continue
        feature, label_match = labeled
        if feature in extracted:
            continue
        suffix = _result_region(line[label_match.end() :])
        value = _first_measurement_number(suffix)
        if value is not None:
            extracted[feature] = ExtractedMeasurement(feature, value, raw_line.strip())
    return extracted


def read_uploaded_report(file: BinaryIO, filename: str) -> str:
    """Read selectable PDF text or OCR an image; uploaded bytes are not persisted."""
    content = file.read()
    suffix = filename.rsplit(".", 1)[-1].lower()
    if suffix in {"txt", "csv"}:
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("Text reports must use UTF-8 encoding.") from error
    if suffix == "pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        if len(reader.pages) > 10:
            raise ValueError("PDFs are limited to 10 pages.")
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        if not text.strip():
            raise ValueError("This PDF has no selectable text. Upload a clear image instead.")
        return text
    import pytesseract
    from PIL import Image

    image = Image.open(io.BytesIO(content)).convert("RGB")
    if image.width * image.height > 25_000_000:
        raise ValueError("Image resolution is too large; use an image below 25 megapixels.")
    return pytesseract.image_to_string(image)
