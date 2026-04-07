"""
ocr.py
------
Text extraction from drawing pages.

Strategy:
  - pdf_vector pages: extract embedded text via PyMuPDF (already in PageData.pdf_text)
  - pdf_raster / tiff pages: use pytesseract on the preprocessed binary image

Returns an OcrResult with:
  - raw_text         : full extracted text
  - regions          : list of word-level bounding boxes {text, x, y, w, h}
  - dimensions       : list of detected "NNN mm" style strings
  - hole_annotations : list of hole-spec strings ("Ø12", "M16", "4x Ø8", …)
  - weld_annotations : list of weld-spec strings ("a=5", "umlaufend", "Kehlnaht", …)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List

import cv2
import numpy as np

from input_loader import PageData
from preprocessing import PreprocessResult, enhance_for_ocr

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns for annotation extraction
# ---------------------------------------------------------------------------

_DIMENSION_RE = re.compile(
    r"(?<!\w)"
    r"(\d{1,5}(?:[.,]\d{1,3})?)"
    r"\s*"
    r"(mm|cm|m)\b",
    re.IGNORECASE,
)

_HOLE_RE = re.compile(
    r"(?:"
    r"(\d+)\s*[xX×]\s*"       # multiplier, e.g. "4x"
    r")?"
    r"(?:"
    r"[ØøOÒÓ∅Φ]\s*(\d+(?:[.,]\d+)?)"  # Ø12 or Ø12.5
    r"|M\s*(\d+(?:[.,]\d+)?)"           # M16
    r")",
    re.IGNORECASE,
)

_WELD_RE = re.compile(
    r"(?:"
    r"a\s*=\s*(\d+(?:[.,]\d+)?)"       # a=5
    r"|s\s*=\s*(\d+(?:[.,]\d+)?)"      # s=5
    r"|z\s*=\s*(\d+(?:[.,]\d+)?)"      # z=5
    r"|\b(umlaufend|allround|kehlnaht|kehlnähte|stumpfnaht|HV|HY|½V|K)\b"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class OcrResult:
    raw_text: str = ""
    regions: List[dict] = field(default_factory=list)       # [{text, x, y, w, h}]
    dimensions: List[str] = field(default_factory=list)
    hole_annotations: List[str] = field(default_factory=list)
    weld_annotations: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_text(page: PageData, pre: PreprocessResult, cfg: dict) -> OcrResult:
    """
    Extract and parse text from the page.

    For vector PDFs the embedded text is used; raster images fall back to
    pytesseract.
    """
    ocr_cfg = cfg.get("ocr", {})

    if page.source_type == "pdf_vector" and page.pdf_text.strip():
        raw_text = page.pdf_text
        # Use word-level bboxes extracted from the PDF (populated in input_loader)
        regions: list[dict] = list(page.pdf_words)
        logger.debug(
            "Page %d: using embedded PDF text (%d chars, %d word regions)",
            page.page_index, len(raw_text), len(regions),
        )
    else:
        raw_text, regions = _run_tesseract(pre.gray, ocr_cfg, page.image)
        logger.debug("Page %d: OCR extracted %d chars", page.page_index, len(raw_text))

    result = OcrResult(raw_text=raw_text, regions=regions)
    _parse_annotations(result)
    return result


# ---------------------------------------------------------------------------
# Tesseract wrapper
# ---------------------------------------------------------------------------

def _run_tesseract(gray: np.ndarray, ocr_cfg: dict, original_bgr: np.ndarray) -> tuple[str, list[dict]]:
    try:
        import pytesseract
    except ImportError as exc:
        raise ImportError(
            "pytesseract is required for raster OCR. Install via: pip install pytesseract"
        ) from exc

    # Windows: set path to tesseract.exe if not in system PATH
    import os
    tess_cmd = ocr_cfg.get("tesseract_cmd")
    if tess_cmd:
        pytesseract.pytesseract.tesseract_cmd = tess_cmd
        logger.debug("Tesseract path from config: %s", tess_cmd)
    elif not _tesseract_in_path():
        # Common Windows default install location
        default = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default):
            pytesseract.pytesseract.tesseract_cmd = default
            logger.debug("Tesseract auto-detected at: %s", default)
        else:
            raise RuntimeError(
                "Tesseract not found. Set 'ocr.tesseract_cmd' in config.json to the full path "
                r"of tesseract.exe, e.g.: C:\Program Files\Tesseract-OCR\tesseract.exe"
            )

    lang = ocr_cfg.get("tesseract_lang", "deu+eng")
    tess_cfg = ocr_cfg.get("tesseract_config", "--psm 11")

    enhanced = enhance_for_ocr(gray, {"ocr": ocr_cfg})

    raw_text = pytesseract.image_to_string(enhanced, lang=lang, config=tess_cfg)

    # Word-level bounding boxes for debug overlay
    regions: list[dict] = []
    try:
        data = pytesseract.image_to_data(
            enhanced, lang=lang, config=tess_cfg,
            output_type=pytesseract.Output.DICT,
        )
        n = len(data["text"])
        for i in range(n):
            word = str(data["text"][i]).strip()
            conf = int(data["conf"][i])
            if word and conf > 0:
                regions.append({
                    "text": word,
                    "x": int(data["left"][i]),
                    "y": int(data["top"][i]),
                    "w": int(data["width"][i]),
                    "h": int(data["height"][i]),
                    "conf": conf,
                })
    except Exception as exc:
        logger.debug("Could not extract word bboxes: %s", exc)

    return raw_text, regions


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def _parse_annotations(result: OcrResult) -> None:
    text = result.raw_text

    # Dimensions
    for m in _DIMENSION_RE.finditer(text):
        result.dimensions.append(m.group(0).strip())

    # Hole annotations
    for m in _HOLE_RE.finditer(text):
        result.hole_annotations.append(m.group(0).strip())

    # Weld annotations
    for m in _WELD_RE.finditer(text):
        result.weld_annotations.append(m.group(0).strip())

    # De-duplicate while preserving order
    result.dimensions = _dedup(result.dimensions)
    result.hole_annotations = _dedup(result.hole_annotations)
    result.weld_annotations = _dedup(result.weld_annotations)

    logger.debug(
        "Annotations – dims: %d, holes: %d, welds: %d",
        len(result.dimensions), len(result.hole_annotations), len(result.weld_annotations),
    )


def _dedup(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in lst:
        key = item.lower().replace(" ", "")
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _tesseract_in_path() -> bool:
    import shutil
    return shutil.which("tesseract") is not None
