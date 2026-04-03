"""
calculation.py
--------------
Manufacturing time estimation from classified geometry.

Configurable parameters (from config.json):
  weld_time_per_mm  : minutes per mm of weld (default 0.02)
  drill_time_per_hole: minutes per drilled hole (default 30 s = 0.5 min)
  setup_time_min    : fixed setup time per job (default 15 min)
  overhead_factor   : multiplier for non-productive time (default 1.1)

All inputs are per-page; the engine aggregates across pages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

from classification import WeldClassification, HoleClassification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-page intermediate result
# ---------------------------------------------------------------------------

@dataclass
class PageCalculation:
    page_index: int
    weld_length_mm: Optional[float]
    weld_length_px: float
    hole_count: int
    hole_diameters_mm: List[float]
    weld_time_min: float
    drill_time_min: float
    page_total_min: float


# ---------------------------------------------------------------------------
# Final aggregated result
# ---------------------------------------------------------------------------

@dataclass
class CalculationResult:
    pages: List[PageCalculation] = field(default_factory=list)
    total_weld_length_mm: Optional[float] = None
    total_weld_length_px: float = 0.0
    total_hole_count: int = 0
    all_diameters_mm: List[float] = field(default_factory=list)
    weld_time_min: float = 0.0
    drill_time_min: float = 0.0
    setup_time_min: float = 0.0
    total_time_min: float = 0.0
    scale_was_available: bool = False


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def calculate(
    weld: WeldClassification,
    holes: HoleClassification,
    page_index: int,
    cfg: dict,
) -> PageCalculation:
    """Compute time estimates for a single page."""
    ccfg = cfg.get("calculation", {})
    weld_rate = float(ccfg.get("weld_time_per_mm", 0.02))    # min/mm
    drill_rate = float(ccfg.get("drill_time_per_hole", 0.5))  # min/hole

    weld_len_mm = weld.total_length_mm
    if weld_len_mm is not None:
        weld_time = weld_len_mm * weld_rate
    else:
        # Fallback: use pixel length with a rough estimate (warn user)
        weld_time = 0.0
        logger.warning(
            "Page %d: no mm scale available; weld time set to 0", page_index
        )

    drill_time = holes.count * drill_rate
    page_total = weld_time + drill_time

    logger.debug(
        "Page %d calc: weld_len=%.1f mm, holes=%d, weld_t=%.2f min, drill_t=%.2f min",
        page_index, weld_len_mm or 0.0, holes.count, weld_time, drill_time,
    )

    return PageCalculation(
        page_index=page_index,
        weld_length_mm=weld_len_mm,
        weld_length_px=weld.total_length_px,
        hole_count=holes.count,
        hole_diameters_mm=list(holes.diameters_mm),
        weld_time_min=weld_time,
        drill_time_min=drill_time,
        page_total_min=page_total,
    )


def aggregate(page_results: list[PageCalculation], cfg: dict) -> CalculationResult:
    """Aggregate per-page calculations into a single job result."""
    ccfg = cfg.get("calculation", {})
    setup = float(ccfg.get("setup_time_min", 15))
    overhead = float(ccfg.get("overhead_factor", 1.1))

    total_weld_mm: Optional[float] = None
    total_weld_px = 0.0
    total_holes = 0
    all_diams: list[float] = []
    weld_time_total = 0.0
    drill_time_total = 0.0
    has_mm_scale = False

    for p in page_results:
        if p.weld_length_mm is not None:
            total_weld_mm = (total_weld_mm or 0.0) + p.weld_length_mm
            has_mm_scale = True
        total_weld_px += p.weld_length_px
        total_holes += p.hole_count
        weld_time_total += p.weld_time_min
        drill_time_total += p.drill_time_min
        for d in p.hole_diameters_mm:
            if d not in all_diams:
                all_diams.append(d)

    all_diams.sort()

    productive_time = (weld_time_total + drill_time_total) * overhead
    total_time = setup + productive_time

    logger.info(
        "Aggregated: weld=%.1f mm, holes=%d, setup=%.0f min, total=%.1f min",
        total_weld_mm or 0.0, total_holes, setup, total_time,
    )

    return CalculationResult(
        pages=page_results,
        total_weld_length_mm=round(total_weld_mm, 2) if total_weld_mm is not None else None,
        total_weld_length_px=round(total_weld_px, 2),
        total_hole_count=total_holes,
        all_diameters_mm=all_diams,
        weld_time_min=round(weld_time_total, 2),
        drill_time_min=round(drill_time_total, 2),
        setup_time_min=setup,
        total_time_min=round(total_time, 2),
        scale_was_available=has_mm_scale,
    )
