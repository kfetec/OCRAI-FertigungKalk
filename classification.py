"""
classification.py
-----------------
Rule-based classification of detected geometry into:
  - Weld seams   (filtered line segments)
  - Drill holes  (filtered circles)

Uses proximity to OCR/LLM annotations and geometric rules to filter
noise from true manufacturing features.

Returns:
  WeldClassification  – list of classified weld segments + cluster info
  HoleClassification  – list of classified holes + merged diameter list
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import List, Optional

from ocr import OcrResult
from scaling import apply_scale

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class WeldSegment:
    x0: float
    y0: float
    x1: float
    y1: float
    length_px: float
    length_mm: Optional[float]
    angle_deg: float
    cluster_id: int = -1


@dataclass
class WeldClassification:
    segments: List[WeldSegment] = field(default_factory=list)
    total_length_px: float = 0.0
    total_length_mm: Optional[float] = None
    cluster_count: int = 0


@dataclass
class HoleInfo:
    cx: float
    cy: float
    radius_px: float
    diameter_px: float
    diameter_mm: Optional[float]


@dataclass
class HoleClassification:
    holes: List[HoleInfo] = field(default_factory=list)
    count: int = 0
    diameters_mm: List[float] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_welds(
    lines: list[dict],
    ocr: OcrResult,
    llm_result: dict,
    scale_mm_per_px: Optional[float],
    cfg: dict,
) -> WeldClassification:
    """
    Filter and classify line segments as weld seams.

    Heuristics:
      1. Minimum length threshold
      2. Proximity to weld annotation bounding boxes
      3. If no annotations found, use all lines above min_length
    """
    ccfg = cfg.get("classification", {})
    min_weld_mm = float(ccfg.get("min_weld_length_mm", 5))
    proximity_px = float(ccfg.get("weld_proximity_px", 80))
    cluster_gap_mm = float(ccfg.get("cluster_gap_mm", 20))

    # Convert min_weld_mm to pixels for pre-filtering
    min_px = (min_weld_mm / scale_mm_per_px) if scale_mm_per_px else 20.0

    # Filter by minimum length
    candidates = [seg for seg in lines if seg["length_px"] >= min_px]

    # Optionally filter by proximity to weld annotation regions
    weld_regions = [r for r in ocr.regions if _text_is_weld_annotation(r["text"])]

    if weld_regions:
        candidates = _filter_by_proximity(candidates, weld_regions, proximity_px)
        logger.debug(
            "Weld proximity filter: %d → %d candidates (from %d annotation regions)",
            len(lines), len(candidates), len(weld_regions),
        )
    else:
        logger.debug(
            "No weld annotation regions found; keeping all %d length-filtered lines",
            len(candidates),
        )

    # Build WeldSegment objects
    segments: list[WeldSegment] = []
    for seg in candidates:
        length_px = seg["length_px"]
        length_mm = apply_scale(length_px, scale_mm_per_px)
        segments.append(WeldSegment(
            x0=seg["x0"], y0=seg["y0"],
            x1=seg["x1"], y1=seg["y1"],
            length_px=length_px,
            length_mm=length_mm,
            angle_deg=seg["angle_deg"],
        ))

    # Cluster connected/nearby segments
    cluster_gap_px = (cluster_gap_mm / scale_mm_per_px) if scale_mm_per_px else 50.0
    _assign_clusters(segments, cluster_gap_px)
    cluster_count = max((s.cluster_id for s in segments), default=-1) + 1

    total_px = sum(s.length_px for s in segments)
    total_mm = apply_scale(total_px, scale_mm_per_px)

    logger.info(
        "Welds classified: %d segments, %d clusters, total=%.1f px (%.1f mm)",
        len(segments), cluster_count, total_px, total_mm or 0.0,
    )

    return WeldClassification(
        segments=segments,
        total_length_px=total_px,
        total_length_mm=total_mm,
        cluster_count=cluster_count,
    )


def classify_holes(
    circles: list[dict],
    ocr: OcrResult,
    llm_result: dict,
    scale_mm_per_px: Optional[float],
    cfg: dict,
) -> HoleClassification:
    """
    Build the final hole list, merging CV-detected circles with
    OCR/LLM annotations.

    Priority:
      1. LLM result (most structured)
      2. OCR hole annotations parsed locally
      3. Raw CV circle detections
    """
    # ---- LLM-derived holes ----
    llm_count: Optional[int] = llm_result.get("hole_count")
    llm_diameters: Optional[list[float]] = llm_result.get("hole_diameters")

    # ---- CV-detected holes ----
    cv_holes: list[HoleInfo] = []
    for c in circles:
        d_mm = apply_scale(c["diameter_px"], scale_mm_per_px)
        cv_holes.append(HoleInfo(
            cx=c["cx"], cy=c["cy"],
            radius_px=c["radius_px"],
            diameter_px=c["diameter_px"],
            diameter_mm=round(d_mm, 2) if d_mm is not None else None,
        ))

    # Determine final count
    final_count = llm_count if llm_count is not None else len(cv_holes)

    # Determine final diameters
    if llm_diameters:
        final_diameters = [round(d, 2) for d in llm_diameters]
    else:
        # Gather from CV detections (round to nearest 0.5 mm)
        cv_diams = [h.diameter_mm for h in cv_holes if h.diameter_mm is not None]
        final_diameters = _cluster_diameters(cv_diams)

    # If LLM gave more holes than CV detected, synthesise placeholder entries
    holes = list(cv_holes)
    if len(holes) < final_count:
        for _ in range(final_count - len(holes)):
            holes.append(HoleInfo(
                cx=0, cy=0,
                radius_px=0, diameter_px=0,
                diameter_mm=final_diameters[0] if final_diameters else None,
            ))

    logger.info(
        "Holes classified: count=%d, diameters=%s",
        final_count, final_diameters,
    )

    return HoleClassification(
        holes=holes[:final_count],
        count=final_count,
        diameters_mm=final_diameters,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _text_is_weld_annotation(text: str) -> bool:
    t = text.lower()
    keywords = ["kehl", "naht", "nähte", "weld", "umlauf", "a=", "s=", "hv", "hy"]
    return any(k in t for k in keywords)


def _filter_by_proximity(
    candidates: list[dict],
    annotation_regions: list[dict],
    proximity_px: float,
) -> list[dict]:
    """Keep line segments that have their midpoint within proximity_px of any annotation."""
    result = []
    for seg in candidates:
        mx = (seg["x0"] + seg["x1"]) / 2
        my = (seg["y0"] + seg["y1"]) / 2
        for region in annotation_regions:
            rx = region["x"] + region["w"] / 2
            ry = region["y"] + region["h"] / 2
            if math.hypot(mx - rx, my - ry) <= proximity_px:
                result.append(seg)
                break
    return result


def _assign_clusters(segments: list[WeldSegment], gap_px: float) -> None:
    """
    Assign cluster IDs to weld segments using single-linkage clustering
    based on endpoint proximity.
    """
    if not segments:
        return

    cluster_id = 0
    unassigned = list(range(len(segments)))
    segments[unassigned[0]].cluster_id = cluster_id

    while unassigned:
        changed = True
        while changed:
            changed = False
            for i in list(unassigned):
                si = segments[i]
                if si.cluster_id != -1:
                    unassigned.remove(i)
                    continue
                for j, sj in enumerate(segments):
                    if sj.cluster_id == -1:
                        continue
                    if _endpoints_close(si, sj, gap_px):
                        si.cluster_id = sj.cluster_id
                        unassigned.remove(i)
                        changed = True
                        break

        if unassigned:
            cluster_id += 1
            segments[unassigned[0]].cluster_id = cluster_id


def _endpoints_close(a: WeldSegment, b: WeldSegment, gap: float) -> bool:
    for ax, ay in [(a.x0, a.y0), (a.x1, a.y1)]:
        for bx, by in [(b.x0, b.y0), (b.x1, b.y1)]:
            if math.hypot(ax - bx, ay - by) <= gap:
                return True
    return False


def _cluster_diameters(diameters_mm: list[float], tol_mm: float = 1.0) -> list[float]:
    """Return unique diameter values, clustering values within tol_mm of each other."""
    if not diameters_mm:
        return []
    unique: list[float] = []
    for d in sorted(diameters_mm):
        for u in unique:
            if abs(d - u) <= tol_mm:
                break
        else:
            unique.append(round(d, 1))
    return unique
