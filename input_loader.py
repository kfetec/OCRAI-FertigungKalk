"""
input_loader.py
---------------
Detects file type (PDF / TIFF) and returns a list of PageData objects.

Each PageData contains:
  - image        : np.ndarray (BGR, full-resolution)
  - page_index   : int
  - source_type  : "pdf_vector" | "pdf_raster" | "tiff"
  - vector_lines : list[dict]   (populated for pdf_vector only)
  - vector_circles: list[dict]  (populated for pdf_vector only)
  - pdf_text     : str          (populated for pdf pages)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
    pages: List[PageData] = []

    doc = fitz.open(str(path))
    logger.info("Opened PDF '%s' with %d page(s)", path.name, len(doc))

    for page_idx, page in enumerate(doc):
        vector_lines, vector_circles = _extract_pdf_vector_geometry(page)
        pdf_text = page.get_text("text")

        has_vector = bool(vector_lines or vector_circles)
        source_type = "pdf_vector" if has_vector else "pdf_raster"

        # Always render to image so downstream steps (OCR, Hough) have pixels
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        # PyMuPDF returns RGB; convert to BGR for OpenCV
        import cv2
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

        pages.append(PageData(
            image=img_bgr,
            page_index=page_idx,
            source_type=source_type,
            vector_lines=vector_lines,
            vector_circles=vector_circles,
            pdf_text=pdf_text,
        ))
        logger.debug(
            "Page %d: source=%s, vector_lines=%d, vector_circles=%d, text_len=%d",
            page_idx, source_type, len(vector_lines), len(vector_circles), len(pdf_text),
        )

    doc.close()
    return pages


def _extract_pdf_vector_geometry(page) -> tuple[list[dict], list[dict]]:
    """
    Parse DrawingPath objects from a PDF page.
    Returns (lines, circles) where each entry is a dict with geometry data
    in *PDF coordinate space* (points, origin bottom-left).
    """
    lines: list[dict] = []
    circles: list[dict] = []

    try:
        paths = page.get_drawings()
    except Exception:
        return lines, circles

    for path in paths:
        for item in path.get("items", []):
            kind = item[0]

            if kind == "l":  # line segment
                p0, p1 = item[1], item[2]
                length = float(((p1.x - p0.x) ** 2 + (p1.y - p0.y) ** 2) ** 0.5)
                lines.append({
                    "x0": float(p0.x), "y0": float(p0.y),
                    "x1": float(p1.x), "y1": float(p1.y),
                    "length_pt": length,
                })

            elif kind == "c":  # cubic Bezier – approximate as circle/arc
                # Rough circle detection: check if bounding rect is square-ish
                rect = path.get("rect")
                if rect and abs((rect.width - rect.height) / max(rect.width, rect.height, 1)) < 0.15:
                    cx = (rect.x0 + rect.x1) / 2
                    cy = (rect.y0 + rect.y1) / 2
                    r = (rect.width + rect.height) / 4
                    circles.append({
                        "cx": float(cx), "cy": float(cy),
                        "radius_pt": float(r),
                    })

    # De-duplicate circles that are very close (same circle split into multiple paths)
    circles = _dedup_circles(circles, tol=2.0)
    return lines, circles


def _dedup_circles(circles: list[dict], tol: float) -> list[dict]:
    unique: list[dict] = []
    for c in circles:
        for u in unique:
            if abs(c["cx"] - u["cx"]) < tol and abs(c["cy"] - u["cy"]) < tol:
                break
        else:
            unique.append(c)
    return unique


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
