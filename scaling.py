"""
scaling.py  ⚠️ CRITICAL
-----------
Compute the scale factor (mm per pixel) for a drawing page.

Algorithm:
  1. Search OCR text for dimension annotations ("NNN mm", "NNN cm", …)
  2. For each found dimension, look for a dimension line near the text bbox
     in the edge map or line list
  3. Compute scale_mm_per_pixel = real_length_mm / pixel_length
  4. Use median of all found scales for robustness

Fallback: if no scale can be determined, return None (downstream code must
handle this gracefully by working in pixels or skipping mm output).
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
# Regex: parse numeric value + unit from a dimension string
# ---------------------------------------------------------------------------

_DIM_PARSE_RE = re.compile(
    r"(\d+(?:[.,]\d+)?)\s*(mm|cm|m)\b",
    re.IGNORECASE,
)

_MM_PER_UNIT = {"mm": 1.0, "cm": 10.0, "m": 1000.0}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_scale(
    ocr: OcrResult,
    lines: list[dict],
    edges: np.ndarray,
    cfg: dict,
) -> Optional[float]:
    """
    Compute scale_mm_per_pixel.

    Parameters
    ----------
    ocr   : OcrResult  containing regions with bbox info
    lines : list[dict] detected line segments (pixel coords)
    edges : np.ndarray Canny edge image (full-res)
    cfg   : dict       top-level config

    Returns
    -------
    float or None  (mm per pixel)
    """
    scfg = cfg.get("scale_detection", {})
    search_radius = float(scfg.get("dim_line_search_radius_px", 50))

    scales: list[float] = []

    for region in ocr.regions:
        text = region.get("text", "")
        m = _DIM_PARSE_RE.search(text)
        if not m:
            continue

        value_str = m.group(1).replace(",", ".")
        unit = m.group(2).lower()
        real_mm = float(value_str) * _MM_PER_UNIT.get(unit, 1.0)

        if real_mm <= 0:
            continue

        # Bounding box of this text region
        rx, ry, rw, rh = region["x"], region["y"], region["w"], region["h"]
        cx_text = rx + rw / 2
        cy_text = ry + rh / 2

        # Find dimension line near this text region
        px_len = _find_dimension_line_near(
            cx_text, cy_text,
            lines, edges,
            search_radius,
        )
        if px_len is None or px_len < 5:
            continue

        scale = real_mm / px_len
        scales.append(scale)
        logger.debug(
            "Scale candidate from '%s': %.2f mm / %d px = %.4f mm/px",
            text.strip(), real_mm, px_len, scale,
        )

    if not scales:
        # Try dimension strings without bounding boxes (e.g. from vector PDF text)
        scales = _scale_from_raw_text(ocr.dimensions, lines, edges, search_radius)

    if not scales:
        logger.warning("Could not determine scale – working in pixels")
        return None

    result = float(np.median(scales))
    logger.info("Scale determined: %.5f mm/px (from %d sample(s))", result, len(scales))
    return result


def apply_scale(value_px: float, scale: Optional[float]) -> Optional[float]:
    """Convert a pixel measurement to mm. Returns None if scale is unknown."""
    if scale is None:
        return None
    return value_px * scale


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_dimension_line_near(
    cx: float,
    cy: float,
    lines: list[dict],
    edges: np.ndarray,
    radius: float,
) -> Optional[float]:
    """
    Return the length (pixels) of the closest horizontal/vertical line
    near the given text centre point.

    First tries the extracted line list; falls back to edge-image scan.
    """
    # 1. Try extracted lines
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

    # 2. Fallback: scan edge image in a bounding strip
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

    # Run HoughLinesP on the strip
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


def _scale_from_raw_text(
    dimension_strings: list[str],
    lines: list[dict],
    edges: np.ndarray,
    radius: float,
) -> list[float]:
    """
    When we have dimension strings but no bbox info (vector PDF), try to
    match against the longest lines heuristically.
    """
    if not dimension_strings or not lines:
        return []

    # Sort lines by length descending
    sorted_lines = sorted(lines, key=lambda s: s["length_px"], reverse=True)
    top_lines = sorted_lines[:min(10, len(sorted_lines))]

    scales: list[float] = []
    for dim_str in dimension_strings:
        m = _DIM_PARSE_RE.search(dim_str)
        if not m:
            continue
        value_str = m.group(1).replace(",", ".")
        unit = m.group(2).lower()
        real_mm = float(value_str) * _MM_PER_UNIT.get(unit, 1.0)
        if real_mm <= 0:
            continue

        for seg in top_lines:
            px_len = seg["length_px"]
            if px_len < 10:
                continue
            candidate = real_mm / px_len
            # Sanity: typical drawing scales 1:1 to 1:100 → 0.01 – 10 mm/px
            if 0.005 <= candidate <= 50.0:
                scales.append(candidate)
                break

    return scales
