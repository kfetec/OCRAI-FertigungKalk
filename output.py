"""
output.py
---------
Formats the final CalculationResult into a structured JSON-serialisable dict
and provides helpers for writing it to disk or printing it.

Output schema:
{
  "input_file": "...",
  "pages_processed": 2,
  "scale_available": true,
  "welds": {
    "total_length_mm": 12500.0,      // null if scale unavailable
    "total_length_px": 4166.7,
    "segment_count": 48,
    "cluster_count": 12,
    "time_min": 250.0
  },
  "holes": {
    "count": 24,
    "diameters_mm": [12.0, 16.0],
    "time_min": 12.0
  },
  "time": {
    "weld_min": 250.0,
    "drill_min": 12.0,
    "setup_min": 15.0,
    "total_min": 304.1,
    "total_hours": 5.07
  },
  "per_page": [ ... ]
}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from calculation import CalculationResult
from classification import WeldClassification, HoleClassification

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_output(
    calc: CalculationResult,
    weld_per_page: list[WeldClassification],
    holes_per_page: list[HoleClassification],
    input_file: str,
) -> dict:
    """
    Assemble the final output dict.

    Parameters
    ----------
    calc            : aggregated CalculationResult
    weld_per_page   : one WeldClassification per page (for segment counts)
    holes_per_page  : one HoleClassification per page
    input_file      : original input file path (for metadata)

    Returns
    -------
    dict  JSON-serialisable output
    """
    total_segments = sum(len(w.segments) for w in weld_per_page)
    total_clusters = sum(w.cluster_count for w in weld_per_page)

    per_page = []
    for p in calc.pages:
        per_page.append({
            "page": p.page_index + 1,
            "weld_length_mm": p.weld_length_mm,
            "weld_length_px": round(p.weld_length_px, 2),
            "hole_count": p.hole_count,
            "hole_diameters_mm": p.hole_diameters_mm,
            "time_min": round(p.page_total_min, 2),
        })

    output = {
        "input_file": str(input_file),
        "pages_processed": len(calc.pages),
        "scale_available": calc.scale_was_available,
        "welds": {
            "total_length_mm": calc.total_weld_length_mm,
            "total_length_px": calc.total_weld_length_px,
            "segment_count": total_segments,
            "cluster_count": total_clusters,
            "time_min": calc.weld_time_min,
        },
        "holes": {
            "count": calc.total_hole_count,
            "diameters_mm": calc.all_diameters_mm,
            "time_min": calc.drill_time_min,
        },
        "time": {
            "weld_min": calc.weld_time_min,
            "drill_min": calc.drill_time_min,
            "setup_min": calc.setup_time_min,
            "total_min": calc.total_time_min,
            "total_hours": round(calc.total_time_min / 60, 3),
        },
        "per_page": per_page,
    }

    return output


def write_json(data: dict, output_path: str | Path) -> None:
    """Write the output dict to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Output written to: %s", path)


def print_summary(data: dict) -> None:
    """Print a human-readable summary to stdout."""
    lines = [
        "",
        "=" * 60,
        f"  OCRAI-FertigungKalk  –  Result Summary",
        "=" * 60,
        f"  File       : {data.get('input_file', 'N/A')}",
        f"  Pages      : {data.get('pages_processed', 0)}",
        f"  Scale      : {'available' if data.get('scale_available') else 'NOT AVAILABLE (pixel units only)'}",
        "",
        "  ── Welds ──────────────────────────────────────────",
    ]

    w = data.get("welds", {})
    weld_len = w.get("total_length_mm")
    if weld_len is not None:
        lines.append(f"  Total length : {weld_len:,.1f} mm")
    else:
        lines.append(f"  Total length : {w.get('total_length_px', 0):,.1f} px (no scale)")
    lines.append(f"  Segments     : {w.get('segment_count', 0)}")
    lines.append(f"  Clusters     : {w.get('cluster_count', 0)}")

    lines += [
        "",
        "  ── Holes ──────────────────────────────────────────",
    ]
    h = data.get("holes", {})
    lines.append(f"  Count        : {h.get('count', 0)}")
    diams = h.get("diameters_mm", [])
    if diams:
        diam_str = ", ".join(f"Ø{d}" for d in diams)
        lines.append(f"  Diameters    : {diam_str} mm")

    t = data.get("time", {})
    lines += [
        "",
        "  ── Estimated Time ─────────────────────────────────",
        f"  Weld         : {t.get('weld_min', 0):.1f} min",
        f"  Drilling     : {t.get('drill_min', 0):.1f} min",
        f"  Setup        : {t.get('setup_min', 0):.1f} min",
        f"  TOTAL        : {t.get('total_min', 0):.1f} min  ({t.get('total_hours', 0):.2f} h)",
        "=" * 60,
        "",
    ]
    print("\n".join(lines))
