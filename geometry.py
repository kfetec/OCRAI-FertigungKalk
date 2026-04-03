"""
geometry.py
-----------
Geometry extraction from both raster images (Hough transforms) and
vector PDF data.

Outputs normalised geometry in *pixel space* (full-resolution).
Scaling to mm happens later in scaling.py.

Structures returned:
  LineSegment  – dict with x0,y0,x1,y1,length_px,angle_deg
  CircleGeom   – dict with cx,cy,radius_px,diameter_px
"""

from __future__ import annotations

import logging
import math
from typing import List

import cv2
import numpy as np

from input_loader import PageData
from preprocessing import PreprocessResult

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_geometry(
    page: PageData,
    pre: PreprocessResult,
    cfg: dict,
) -> tuple[list[dict], list[dict]]:
    """
    Return (lines, circles) for the given page.

    For pdf_vector pages, vector data is preferred and Hough is used only
    as a supplement when vector data is sparse.

    Parameters
    ----------
    page : PageData       from input_loader
    pre  : PreprocessResult from preprocessing
    cfg  : dict           top-level config

    Returns
    -------
    lines   : list of LineSegment dicts (pixel coords, full-res)
    circles : list of CircleGeom dicts  (pixel coords, full-res)
    """
    if page.source_type == "pdf_vector" and (page.vector_lines or page.vector_circles):
        lines = _lines_from_vector(page, cfg)
        circles = _circles_from_vector(page, cfg)
        logger.info(
            "Page %d: using vector geometry – %d lines, %d circles",
            page.page_index, len(lines), len(circles),
        )
        # Supplement with Hough when vector data looks incomplete
        if len(lines) < 5:
            logger.debug("Page %d: vector lines sparse, supplementing with Hough", page.page_index)
            hough_lines, hough_circles = _detect_raster(pre, cfg)
            lines = _merge_line_lists(lines, hough_lines, tol_px=10)
            if not circles:
                circles = hough_circles
    else:
        lines, circles = _detect_raster(pre, cfg)
        logger.info(
            "Page %d: using raster geometry – %d lines, %d circles",
            page.page_index, len(lines), len(circles),
        )

    return lines, circles


# ---------------------------------------------------------------------------
# Vector geometry conversion
# ---------------------------------------------------------------------------

def _lines_from_vector(page: PageData, cfg: dict) -> list[dict]:
    """Convert PDF vector lines (pt units) to pixel-space line segments."""
    dpi = cfg.get("pdf_render_dpi", 300)
    pt_to_px = dpi / 72.0

    # PDF coordinate origin is bottom-left; image origin is top-left.
    # We need to flip the y-axis.
    img_h = page.image.shape[0]

    lines = []
    for vl in page.vector_lines:
        x0 = vl["x0"] * pt_to_px
        y0 = img_h - vl["y0"] * pt_to_px
        x1 = vl["x1"] * pt_to_px
        y1 = img_h - vl["y1"] * pt_to_px
        length_px = math.hypot(x1 - x0, y1 - y0)
        angle_deg = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
        lines.append({
            "x0": x0, "y0": y0,
            "x1": x1, "y1": y1,
            "length_px": length_px,
            "angle_deg": angle_deg,
        })

    return _merge_collinear_segments(lines, cfg)


def _circles_from_vector(page: PageData, cfg: dict) -> list[dict]:
    """Convert PDF vector circles (pt units) to pixel-space circles."""
    dpi = cfg.get("pdf_render_dpi", 300)
    pt_to_px = dpi / 72.0
    img_h = page.image.shape[0]

    circles = []
    for vc in page.vector_circles:
        cx = vc["cx"] * pt_to_px
        cy = img_h - vc["cy"] * pt_to_px
        r = vc["radius_pt"] * pt_to_px
        circles.append({
            "cx": cx, "cy": cy,
            "radius_px": r,
            "diameter_px": 2 * r,
        })
    return circles


# ---------------------------------------------------------------------------
# Raster (Hough) detection
# ---------------------------------------------------------------------------

def _detect_raster(pre: PreprocessResult, cfg: dict) -> tuple[list[dict], list[dict]]:
    lines = _detect_lines_hough(pre, cfg)
    circles = _detect_circles_hough(pre, cfg)
    return lines, circles


def _detect_lines_hough(pre: PreprocessResult, cfg: dict) -> list[dict]:
    hcfg = cfg.get("hough_lines", {})
    scale = pre.scale_factor  # pixels in small image → full-res pixels = px / scale

    rho = float(hcfg.get("rho", 1))
    theta = math.radians(float(hcfg.get("theta_deg", 1)))
    threshold = int(hcfg.get("threshold", 80))
    min_len = float(hcfg.get("min_line_length", 30))
    max_gap = float(hcfg.get("max_line_gap", 10))

    raw = cv2.HoughLinesP(
        pre.edges_small,
        rho, theta, threshold,
        minLineLength=min_len,
        maxLineGap=max_gap,
    )

    if raw is None:
        logger.debug("HoughLinesP returned no lines")
        return []

    lines = []
    for seg in raw:
        x0, y0, x1, y1 = (v / scale for v in seg[0])
        length_px = math.hypot(x1 - x0, y1 - y0)
        angle_deg = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
        lines.append({
            "x0": x0, "y0": y0,
            "x1": x1, "y1": y1,
            "length_px": length_px,
            "angle_deg": angle_deg,
        })

    logger.debug("HoughLinesP raw: %d segments", len(lines))
    return _merge_collinear_segments(lines, cfg)


def _detect_circles_hough(pre: PreprocessResult, cfg: dict) -> list[dict]:
    hcfg = cfg.get("hough_circles", {})
    scale = pre.scale_factor

    dp = float(hcfg.get("dp", 1.2))
    min_dist = float(hcfg.get("min_dist_px", 20))
    param1 = float(hcfg.get("param1", 50))
    param2 = float(hcfg.get("param2", 30))
    min_r = int(hcfg.get("min_radius_px", 3))
    max_r = int(hcfg.get("max_radius_px", 200))

    # HoughCircles works best on blurred gray
    blurred_small = cv2.GaussianBlur(pre.gray_small, (9, 9), 2)
    raw = cv2.HoughCircles(
        blurred_small, cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_r,
        maxRadius=max_r,
    )

    if raw is None:
        logger.debug("HoughCircles returned no circles")
        return []

    circles = []
    for cx, cy, r in raw[0]:
        circles.append({
            "cx": float(cx) / scale,
            "cy": float(cy) / scale,
            "radius_px": float(r) / scale,
            "diameter_px": 2 * float(r) / scale,
        })

    logger.debug("HoughCircles detected: %d", len(circles))
    return circles


# ---------------------------------------------------------------------------
# Segment merging helpers
# ---------------------------------------------------------------------------

def _merge_collinear_segments(lines: list[dict], cfg: dict) -> list[dict]:
    """
    Merge nearly-collinear overlapping/touching line segments.
    Uses angle and perpendicular distance criteria.
    """
    hcfg = cfg.get("hough_lines", {})
    dist_tol = float(hcfg.get("merge_distance_px", 5))
    angle_tol = float(hcfg.get("merge_angle_deg", 2.0))

    if not lines:
        return lines

    merged: list[dict] = []
    used = [False] * len(lines)

    for i, seg_i in enumerate(lines):
        if used[i]:
            continue
        group = [seg_i]
        used[i] = True

        for j, seg_j in enumerate(lines):
            if used[j]:
                continue
            if _are_collinear(seg_i, seg_j, angle_tol, dist_tol):
                group.append(seg_j)
                used[j] = True

        merged_seg = _fuse_segment_group(group)
        merged.append(merged_seg)

    logger.debug("Collinear merge: %d → %d segments", len(lines), len(merged))
    return merged


def _are_collinear(a: dict, b: dict, angle_tol: float, dist_tol: float) -> bool:
    angle_diff = abs(a["angle_deg"] - b["angle_deg"])
    angle_diff = min(angle_diff, 180.0 - angle_diff)
    if angle_diff > angle_tol:
        return False
    # Perpendicular distance from midpoint of b to line defined by a
    mx = (b["x0"] + b["x1"]) / 2
    my = (b["y0"] + b["y1"]) / 2
    d = _point_to_line_dist(mx, my, a["x0"], a["y0"], a["x1"], a["y1"])
    return d <= dist_tol


def _point_to_line_dist(px, py, x0, y0, x1, y1) -> float:
    dx, dy = x1 - x0, y1 - y0
    denom = math.hypot(dx, dy)
    if denom < 1e-9:
        return math.hypot(px - x0, py - y0)
    return abs(dy * px - dx * py + x1 * y0 - y1 * x0) / denom


def _fuse_segment_group(group: list[dict]) -> dict:
    """Fuse a group of collinear segments into one spanning segment."""
    if len(group) == 1:
        return group[0]

    # Project all endpoints onto the direction of the first segment
    ref = group[0]
    dx = ref["x1"] - ref["x0"]
    dy = ref["y1"] - ref["y0"]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length

    projections = []
    for seg in group:
        for px, py in [(seg["x0"], seg["y0"]), (seg["x1"], seg["y1"])]:
            t = (px - ref["x0"]) * ux + (py - ref["y0"]) * uy
            projections.append((t, px, py))

    projections.sort(key=lambda v: v[0])
    _, x0, y0 = projections[0]
    _, x1, y1 = projections[-1]
    length_px = math.hypot(x1 - x0, y1 - y0)
    angle_deg = math.degrees(math.atan2(y1 - y0, x1 - x0)) % 180.0
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "length_px": length_px, "angle_deg": angle_deg}


def _merge_line_lists(a: list[dict], b: list[dict], tol_px: float) -> list[dict]:
    """Append lines from b that are not already represented in a."""
    result = list(a)
    for seg in b:
        mx = (seg["x0"] + seg["x1"]) / 2
        my = (seg["y0"] + seg["y1"]) / 2
        duplicate = any(
            math.hypot(mx - (s["x0"] + s["x1"]) / 2, my - (s["y0"] + s["y1"]) / 2) < tol_px
            for s in result
        )
        if not duplicate:
            result.append(seg)
    return result
