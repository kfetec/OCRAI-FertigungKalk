"""
input_loader.py
---------------
Detects file type (PDF / TIFF) and returns a list of PageData objects.

Each PageData contains:
  - image         : np.ndarray (BGR, full-resolution)
  - page_index    : int
  - source_type   : "pdf_vector" | "pdf_raster" | "tiff"
  - vector_lines  : list[dict]   (populated for pdf_vector only)
  - vector_circles: list[dict]   (populated for pdf_vector only)
  - pdf_text      : str          (populated for pdf pages)
  - pdf_words     : list[dict]   word-level bboxes in pixel coords [{text,x,y,w,h}]
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class PageData:
    image: np.ndarray
    page_index: int
    source_type: str  # "pdf_vector" | "pdf_raster" | "tiff"
    vector_lines: List[dict] = field(default_factory=list)
    vector_circles: List[dict] = field(default_factory=list)
    pdf_text: str = ""
    pdf_words: List[dict] = field(default_factory=list)  # [{text, x, y, w, h}]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load(file_path: str | Path, cfg: dict) -> List[PageData]:
    """Load a PDF or TIFF file and return one PageData per page."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path, cfg)
    elif suffix in (".tif", ".tiff"):
        return _load_tiff(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix!r}. Expected .pdf or .tif/.tiff")


# ---------------------------------------------------------------------------
# PDF loader
# ---------------------------------------------------------------------------

def _load_pdf(path: Path, cfg: dict) -> List[PageData]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise ImportError("PyMuPDF is required for PDF processing. Install via: pip install pymupdf") from exc

    dpi = cfg.get("pdf_render_dpi", 300)
    pt_to_px = dpi / 72.0
    pages: List[PageData] = []

    doc = fitz.open(str(path))
    logger.info("Opened PDF '%s' with %d page(s)", path.name, len(doc))

    for page_idx, page in enumerate(doc):
        vector_lines, vector_circles = _extract_pdf_vector_geometry(page, cfg)
        pdf_text = page.get_text("text")

        has_vector = bool(vector_lines or vector_circles)
        source_type = "pdf_vector" if has_vector else "pdf_raster"

        # Always render to image so downstream steps have pixels
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        import cv2
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        img_h = img_bgr.shape[0]

        # Extract word-level bboxes in pixel space for scale detection + debug
        pdf_words = _extract_pdf_words(page, pt_to_px, img_h)

        pages.append(PageData(
            image=img_bgr,
            page_index=page_idx,
            source_type=source_type,
            vector_lines=vector_lines,
            vector_circles=vector_circles,
            pdf_text=pdf_text,
            pdf_words=pdf_words,
        ))
        logger.debug(
            "Page %d: source=%s, vector_lines=%d, vector_circles=%d, words=%d",
            page_idx, source_type, len(vector_lines), len(vector_circles), len(pdf_words),
        )

    doc.close()
    return pages


def _extract_pdf_words(page, pt_to_px: float, img_h: int) -> list[dict]:
    """
    Extract word-level bounding boxes from a PDF page and convert to pixel coords.
    Returns list of {text, x, y, w, h, conf}.
    PDF word format: (x0, y0, x1, y1, word, block_no, line_no, word_no)
    PDF y-axis: origin at TOP-LEFT (in get_text("words")), so no flip needed.
    """
    words = []
    try:
        raw = page.get_text("words")
    except Exception:
        return words

    for entry in raw:
        x0, y0, x1, y1 = entry[0], entry[1], entry[2], entry[3]
        text = str(entry[4]).strip()
        if not text:
            continue
        px = int(x0 * pt_to_px)
        py = int(y0 * pt_to_px)
        pw = max(1, int((x1 - x0) * pt_to_px))
        ph = max(1, int((y1 - y0) * pt_to_px))
        words.append({"text": text, "x": px, "y": py, "w": pw, "h": ph, "conf": 95})

    return words


def _extract_pdf_vector_geometry(page, cfg: dict) -> tuple[list[dict], list[dict]]:
    """
    Parse DrawingPath objects from a PDF page.

    Only extracts LINE segments from vector paths.
    Circle/hole detection is intentionally NOT done here – Bezier-curve analysis
    is too unreliable across CAD exporters.  Holes are detected via:
      - OCR / LLM annotation parsing  (primary)
      - Hough circle transform on the rendered image  (fallback)
    """
    lines: list[dict] = []

    try:
        paths = page.get_drawings()
    except Exception:
        return lines, []

    for path in paths:
        for item in path.get("items", []):
            if item[0] == "l":
                p0, p1 = item[1], item[2]
                length = math.hypot(p1.x - p0.x, p1.y - p0.y)
                if length > 0.1:
                    lines.append({
                        "x0": float(p0.x), "y0": float(p0.y),
                        "x1": float(p1.x), "y1": float(p1.y),
                        "length_pt": length,
                    })

    logger.debug("Vector geometry extracted: %d line segments", len(lines))
    return lines, []  # circles always empty – detected elsewhere


# ---------------------------------------------------------------------------
# TIFF loader
# ---------------------------------------------------------------------------

def _load_tiff(path: Path) -> List[PageData]:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("Pillow is required for TIFF processing. Install via: pip install Pillow") from exc

    import cv2

    pages: List[PageData] = []
    img_pil = Image.open(str(path))

    page_idx = 0
    while True:
        frame = img_pil.copy().convert("RGB")
        arr = np.array(frame, dtype=np.uint8)
        img_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        pages.append(PageData(
            image=img_bgr,
            page_index=page_idx,
            source_type="tiff",
        ))
        logger.debug("TIFF page %d loaded: shape=%s", page_idx, img_bgr.shape)

        page_idx += 1
        try:
            img_pil.seek(page_idx)
        except EOFError:
            break

    logger.info("Loaded TIFF '%s' with %d page(s)", path.name, len(pages))
    return pages
