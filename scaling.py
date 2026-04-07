"""
scaling.py  ⚠️ CRITICAL
-----------
Compute the scale factor (mm per pixel) for a drawing page.

Strategy (in order of priority):
  1. Explicit scale ratio in text ("1:1", "1:2", "1:5", "2:1", …)
     Combined with PDF rendering DPI → exact mm/px
  2. Dimension annotation with unit suffix ("100 mm", "50 cm") near a
     detected dimension line → ratio from real_mm / pixel_length
  3. Pure-number dimension + matching vector line (for drawings without
     explicit "mm" unit labels)

Fallback: None (downstream code works in pixels or omits mm output).
"""

from __future__ import annotations

import logging
import math
import re
from typing import Optional

import cv2
import numpy as np

from ocr import OcrResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# "1:2", "1 : 2", "M 1:50", "MASSTAB 1:5"
_RATIO_RE = re.compile(
    r"(?:ma[sß]{1,2}stab|massstab|scale|m)?\s*"
    r"(\d+(?:[.,]\d+)?)\s*:\s*(\d+(?:[.,]\d+)?)",
    re.IGNORECASE,
)

# "100 mm", "50cm", "1.5m"
_DIM_UNIT_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m)\b",
    re.IGNORECASE,
)

# Pure numbers that look like dimensions (e.g. "178", "45") – used as fallback
_DIM_NUMBER_RE = re.compile(r"\b(\d{1,5}(?:[.,]\d{1,2})?)\b")

_MM_PER_UNIT = {"mm": 1.0, "cm": 10.0, "m": 1000.0}

# 1 PDF point = 25.4/72 mm
_MM_PER_PT = 25.4 / 72.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_scale(
    ocr: OcrResult,
    lines: list[dict],
    edges: np.ndarray,
    cfg: dict,
    source_type: str = "pdf_vector",
    pdf_render_dpi: int = 300,
) -> Optional[float]:
    """
    Compute scale_mm_per_pixel.

    Parameters
    ----------
    ocr              : OcrResult with regions (word bboxes) and raw_text
    lines            : detected line segments in pixel coords
    edges            : Canny edge image (full-res)
    cfg              : top-level config
    source_type      : "pdf_vector", "pdf_raster", or "tiff"
    pdf_render_dpi   : DPI used to render the PDF (needed for DPI-based scale)

    Returns
    -------
    float or None  (mm per pixel)
    """
    scfg = cfg.get("scale_detection", {})
    search_radius = float(scfg.get("dim_line_search_radius_px", 50))

    # ── Strategy 1: explicit scale ratio in text ────────────────────────────
    if source_type in ("pdf_vector", "pdf_raster"):
        scale = _scale_from_ratio(ocr.raw_text, pdf_render_dpi)
        if scale is not None:
            logger.info("Scale from ratio notation: %.5f mm/px", scale)
            return scale

    # ── Strategy 2: dimension with unit + dimension line ────────────────────
    scales: list[float] = []
    for region in ocr.regions:
        text = region.get("text", "")
        m = _DIM_UNIT_RE.search(text)
        if not m:
            continue
        real_mm = float(m.group(1).replace(",", ".")) * _MM_PER_UNIT[m.group(2).lower()]
        if real_mm <= 0:
            continue
        cx = region["x"] + region["w"] / 2
        cy = region["y"] + region["h"] / 2
        px_len = _find_dimension_line_near(cx, cy, lines, edges, search_radius)
        if px_len and px_len >= 5:
            scales.append(real_mm / px_len)
            logger.debug("Scale candidate from '%s': %.4f mm/px", text, real_mm / px_len)

    if scales:
        result = float(np.median(scales))
        logger.info("Scale from dimension annotations: %.5f mm/px", result)
        return result

    # ── Strategy 3: pure-number dimensions matched to vector lines ──────────
    if source_type == "pdf_vector" and lines:
        scale = _scale_from_pure_numbers(ocr, lines)
        if scale is not None:
            logger.info("Scale from pure-number heuristic: %.5f mm/px", scale)
            return scale

    logger.warning("Could not determine scale – working in pixels")
    return None


def apply_scale(value_px: float, scale: Optional[float]) -> Optional[float]:
    """Convert a pixel measurement to mm. Returns None if scale is unknown."""
    if scale is None:
        return None
    return value_px * scale


# ---------------------------------------------------------------------------
# Strategy 1: ratio from text
# ---------------------------------------------------------------------------

def _scale_from_ratio(text: str, dpi: int) -> Optional[float]:
    """
    Look for "1:2", "Maßstab 1:5" etc. in the text.
    Returns mm/px = (drawing_units / real_units) × (25.4 / dpi)
    e.g. "1:1" at 300 DPI → 1 × (25.4/300) = 0.08467 mm/px
         "1:2" at 300 DPI → 2 × (25.4/300) = 0.16933 mm/px
    """
    px_per_inch = float(dpi)
    mm_per_px = 25.4 / px_per_inch

    for m in _RATIO_RE.finditer(text):
        a = float(m.group(1).replace(",", "."))
        b = float(m.group(2).replace(",", "."))
        if a <= 0 or b <= 0:
            continue
        # Drawing scale A:B means 1 drawing unit = (B/A) real units
        # At "1:1": 1mm on paper = 1mm real → mm_per_px = 25.4/dpi
        # At "1:2": 1mm on paper = 2mm real → mm_per_px = 2 × 25.4/dpi
        real_factor = b / a
        # Sanity: typical drawing scales 1:1 to 1:100 (real_factor 0.01–100)
        if 0.01 <= real_factor <= 100.0:
            candidate = mm_per_px * real_factor
            logger.debug(
                "Ratio '%s:%s' → real_factor=%.3f → %.5f mm/px",
                m.group(1), m.group(2), real_factor, candidate,
            )
            return candidate

    return None


# ---------------------------------------------------------------------------
# Strategy 3: pure-number heuristic for vector PDFs
# ---------------------------------------------------------------------------

def _scale_from_pure_numbers(ocr: OcrResult, lines: list[dict]) -> Optional[float]:
    """
    For vector PDFs without explicit unit labels:
    Match labeled dimension numbers to nearby line lengths (in pixels).

    Uses only regions that contain standalone numbers (e.g. "45", "178").
    Skips numbers that look like angles (°), radii (R prefix), or tolerances.
    """
    # Collect numeric regions (word bbox + value)
    candidates: list[tuple[float, float, float, float]] = []  # (real_mm, cx, cy, conf)

    for region in ocr.regions:
        text = region.get("text", "").strip()
        # Skip if it contains non-numeric chars besides comma/dot
        if re.search(r"[°RrMm]", text):
            continue
        m = re.fullmatch(r"(\d{1,5}(?:[.,]\d{1,2})?)", text)
        if not m:
            continue
        value = float(m.group(1).replace(",", "."))
        # Typical engineering dimensions: 5–5000 mm
        if not (5 <= value <= 5000):
            continue
        cx = region["x"] + region["w"] / 2
        cy = region["y"] + region["h"] / 2
        candidates.append((value, cx, cy))

    if not candidates:
        return None

    # For each candidate, find the closest line and compute scale
    search_radius = 80.0
    scales: list[float] = []

    for real_mm, cx, cy in candidates:
        px_len = _find_dimension_line_near(cx, cy, lines, None, search_radius)
        if px_len and px_len >= 5:
            s = real_mm / px_len
            # Sanity: typical drawing renders → 0.01–2 mm/px
            if 0.01 <= s <= 2.0:
                scales.append(s)

    if not scales:
        return None

    # Use median; require at least 2 consistent samples
    if len(scales) < 2:
        return None

    result = float(np.median(scales))
    # Reject if spread is too large (inconsistent matches)
    spread = max(scales) / max(min(scales), 1e-9)
    if spread > 3.0:
        logger.debug("Pure-number scale candidates too spread (%.1fx), discarding", spread)
        return None

    return result


# ---------------------------------------------------------------------------
# Shared helper: find dimension line near text
# ---------------------------------------------------------------------------

def _find_dimension_line_near(
    cx: float,
    cy: float,
    lines: list[dict],
    edges,
    radius: float,
) -> Optional[float]:
    """Return length (px) of the closest line segment near the given point."""
    best_dist = radius
    best_len: Optional[float] = None

    for seg in lines:
        mx = (seg["x0"] + seg["x1"]) / 2
        my = (seg["y0"] + seg["y1"]) / 2
        dist = math.hypot(mx - cx, my - cy)
        if dist < best_dist:
            best_dist = dist
            best_len = seg["length_px"]

    if best_len is not None:
        return best_len

    # Fallback: scan edge image strip
    if edges is None:
        return None

    h, w = edges.shape
    x0 = max(0, int(cx - radius))
    x1 = min(w, int(cx + radius))
    y0 = max(0, int(cy - radius))
    y1 = min(h, int(cy + radius))
    strip = edges[y0:y1, x0:x1]
    if strip.size == 0:
        return None

    local_lines = cv2.HoughLinesP(strip, 1, math.pi / 180, 20, minLineLength=10, maxLineGap=5)
    if local_lines is None:
        return None

    best: Optional[float] = None
    for seg in local_lines:
        lx0, ly0, lx1, ly1 = seg[0]
        length = math.hypot(lx1 - lx0, ly1 - ly0)
        if best is None or length > best:
            best = length

    return best
