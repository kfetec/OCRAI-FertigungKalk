"""
debug.py
--------
Visual debug overlays for each processed page.

Overlays drawn on the original image (BGR):
  Green  – detected / classified weld line segments
  Blue   – detected / classified drill holes (circles)
  Red    – OCR text bounding boxes

Output files:
  {debug_dir}/page_{N+1}.png
  {debug_dir}/page_{N+1}_edges.png   (Canny edge map)
  {debug_dir}/page_{N+1}_thresh.png  (adaptive threshold)

Enabled/disabled via cfg["debug"]["enabled"].
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from classification import WeldClassification, HoleClassification
from ocr import OcrResult
from preprocessing import PreprocessResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_debug_page(
    page_index: int,
    image: np.ndarray,
    pre: PreprocessResult,
    welds: WeldClassification,
    holes: HoleClassification,
    ocr: OcrResult,
    scale_mm_per_px: Optional[float],
    cfg: dict,
) -> None:
    """
    Generate and save debug visualisations for one page.
    Does nothing if cfg["debug"]["enabled"] is False.
    """
    dcfg = cfg.get("debug", {})
    if not dcfg.get("enabled", True):
        return

    out_dir = Path(dcfg.get("output_dir", "debug"))
    out_dir.mkdir(parents=True, exist_ok=True)

    page_num = page_index + 1

    # ---- Main overlay ----
    overlay = image.copy()
    _draw_welds(overlay, welds, dcfg)
    _draw_holes(overlay, holes, dcfg)
    _draw_ocr_regions(overlay, ocr, dcfg)
    _draw_scale_info(overlay, scale_mm_per_px)

    main_path = out_dir / f"page_{page_num}.png"
    cv2.imwrite(str(main_path), overlay)
    logger.info("Debug overlay saved: %s", main_path)

    # ---- Edge map ----
    edges_path = out_dir / f"page_{page_num}_edges.png"
    cv2.imwrite(str(edges_path), pre.edges)

    # ---- Threshold image ----
    thresh_path = out_dir / f"page_{page_num}_thresh.png"
    cv2.imwrite(str(thresh_path), pre.thresh)

    logger.debug("Debug images saved for page %d", page_num)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_welds(img: np.ndarray, welds: WeldClassification, dcfg: dict) -> None:
    color = tuple(dcfg.get("line_color", [0, 255, 0]))  # BGR green
    thickness = int(dcfg.get("line_thickness", 2))

    for seg in welds.segments:
        pt0 = (int(seg.x0), int(seg.y0))
        pt1 = (int(seg.x1), int(seg.y1))
        cv2.line(img, pt0, pt1, color, thickness)

        # Annotate length
        mx = int((seg.x0 + seg.x1) / 2)
        my = int((seg.y0 + seg.y1) / 2)
        if seg.length_mm is not None:
            label = f"{seg.length_mm:.0f}mm"
        else:
            label = f"{seg.length_px:.0f}px"
        cv2.putText(
            img, label, (mx, my - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )


def _draw_holes(img: np.ndarray, holes: HoleClassification, dcfg: dict) -> None:
    color = tuple(dcfg.get("circle_color", [255, 0, 0]))  # BGR blue
    thickness = int(dcfg.get("line_thickness", 2))

    for hole in holes.holes:
        cx = int(hole.cx)
        cy = int(hole.cy)
        r = max(1, int(hole.radius_px))
        if cx == 0 and cy == 0:
            continue  # Synthesised placeholder without position
        cv2.circle(img, (cx, cy), r, color, thickness)
        if hole.diameter_mm is not None:
            label = f"Ø{hole.diameter_mm:.1f}"
        else:
            label = f"Ø{hole.diameter_px:.0f}px"
        cv2.putText(
            img, label, (cx + r + 2, cy),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA,
        )


def _draw_ocr_regions(img: np.ndarray, ocr: OcrResult, dcfg: dict) -> None:
    color = tuple(dcfg.get("ocr_color", [0, 0, 255]))  # BGR red
    thickness = 1

    for region in ocr.regions:
        x, y, w, h = region["x"], region["y"], region["w"], region["h"]
        if w <= 0 or h <= 0:
            continue
        cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
        text = region.get("text", "")[:12]
        cv2.putText(
            img, text, (x, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.28, color, 1, cv2.LINE_AA,
        )


def _draw_scale_info(img: np.ndarray, scale_mm_per_px: Optional[float]) -> None:
    h, w = img.shape[:2]
    if scale_mm_per_px is not None:
        text = f"Scale: {scale_mm_per_px:.4f} mm/px"
        color = (0, 200, 200)  # cyan
    else:
        text = "Scale: UNKNOWN"
        color = (0, 0, 200)   # red-ish

    cv2.putText(
        img, text, (10, h - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
    )
