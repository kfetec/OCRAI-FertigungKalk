"""
main.py
-------
Entry point for the OCRAI-FertigungKalk manufacturing data extraction system.

Pipeline:
  input → load → preprocess → geometry → OCR → scaling
       → hole_detector (LLM vision, 2-stage) → weld classification
       → calculation → output

Usage:
  python main.py drawing.pdf
  python main.py scan.tiff --output results/my_drawing.json
  python main.py drawing.pdf --no-debug --log-level DEBUG
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import input_loader
import preprocessing
import geometry
import ocr as ocr_module
import scaling
import llm_interface
import hole_detector
import classification
import calculation
import output
import debug as debug_module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(input_file: str, cfg: dict) -> dict:
    """Execute the full extraction pipeline. Returns the final output dict."""
    logger.info("=" * 60)
    logger.info("Processing: %s", input_file)
    logger.info("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────────
    pages = input_loader.load(input_file, cfg)
    logger.info("Loaded %d page(s)", len(pages))

    page_calcs: list[calculation.PageCalculation] = []
    weld_per_page: list[classification.WeldClassification] = []
    hole_results_per_page: list[hole_detector.HoleDetectionResult] = []

    all_weld_annotations: list[str] = []
    page_intermediates = []

    # ── 2–5. Per-page: preprocess, geometry, OCR, scale ──────────────────────
    for page in pages:
        logger.info(
            "── Page %d / %d ─────────────────────────────────────",
            page.page_index + 1, len(pages),
        )

        pre = preprocessing.preprocess(page.image, cfg)
        lines, circles = geometry.extract_geometry(page, pre, cfg)
        ocr_result = ocr_module.extract_text(page, pre, cfg)

        scale_mm_per_px = scaling.compute_scale(
            ocr_result, lines, pre.edges, cfg,
            source_type=page.source_type,
            pdf_render_dpi=cfg.get("pdf_render_dpi", 300),
        )

        all_weld_annotations.extend(ocr_result.weld_annotations)
        page_intermediates.append((page, pre, lines, circles, ocr_result, scale_mm_per_px))

    # ── 6. Weld annotation interpretation (text LLM) ─────────────────────────
    logger.info("LLM: interpreting weld annotations …")
    weld_llm = llm_interface.interpret_annotations(
        [],  # holes handled separately by hole_detector
        _dedup(all_weld_annotations),
        cfg,
    )

    # ── 7. Per-page: hole detection + weld classification + calculation ───────
    for page, pre, lines, circles, ocr_result, scale_mm_per_px in page_intermediates:

        # ── 7a. Hole detection (new LLM vision pipeline) ─────────────────────
        logger.info("Page %d: LLM vision hole detection …", page.page_index + 1)
        hole_result = hole_detector.detect_holes(page, ocr_result, cfg)

        # ── 7b. Weld classification ───────────────────────────────────────────
        welds = classification.classify_welds(
            lines, ocr_result, weld_llm, scale_mm_per_px, cfg
        )

        # ── 7c. Build HoleClassification from HoleDetectionResult ─────────────
        holes = _hole_result_to_classification(hole_result, circles, scale_mm_per_px, cfg)

        # ── 7d. Time calculation ──────────────────────────────────────────────
        page_calc = calculation.calculate(welds, holes, page.page_index, cfg)

        weld_per_page.append(welds)
        hole_results_per_page.append(hole_result)
        page_calcs.append(page_calc)

        # ── 7e. Debug overlay ─────────────────────────────────────────────────
        debug_module.save_debug_page(
            page.page_index, page.image, pre,
            welds, holes, ocr_result,
            scale_mm_per_px, cfg,
        )

    # ── 8. Aggregate + output ─────────────────────────────────────────────────
    calc_result = calculation.aggregate(page_calcs, cfg)

    # Rebuild hole_per_page as HoleClassification list for output.build_output
    holes_for_output = [
        _hole_result_to_classification(hr, [], None, cfg)
        for hr in hole_results_per_page
    ]

    result = output.build_output(calc_result, weld_per_page, holes_for_output, input_file)

    # Enrich output with hole detector details
    result["hole_detection"] = _summarise_hole_detections(hole_results_per_page)

    return result


# ---------------------------------------------------------------------------
# Adapter: HoleDetectionResult → HoleClassification
# ---------------------------------------------------------------------------

def _hole_result_to_classification(
    hr: hole_detector.HoleDetectionResult,
    circles: list[dict],
    scale_mm_per_px,
    cfg: dict,
) -> classification.HoleClassification:
    """Convert the new HoleDetectionResult into the legacy HoleClassification."""
    from classification import HoleClassification, HoleInfo
    from scaling import apply_scale

    holes: list[HoleInfo] = []
    for c in circles:
        d_mm = apply_scale(c["diameter_px"], scale_mm_per_px)
        holes.append(HoleInfo(
            cx=c["cx"], cy=c["cy"],
            radius_px=c["radius_px"],
            diameter_px=c["diameter_px"],
            diameter_mm=round(d_mm, 2) if d_mm else None,
        ))

    return HoleClassification(
        holes=holes,
        count=hr.total_count,
        diameters_mm=hr.diameters_mm,
    )


def _summarise_hole_detections(
    results: list[hole_detector.HoleDetectionResult],
) -> dict:
    total = sum(r.total_count for r in results)
    all_diams: list[float] = []
    all_threads: list[str] = []
    total_slots = 0
    per_page = []

    for i, r in enumerate(results):
        for d in r.diameters_mm:
            if d not in all_diams:
                all_diams.append(d)
        for t in r.thread_specs:
            if t not in all_threads:
                all_threads.append(t)
        total_slots += r.slot_count
        per_page.append({
            "page": i + 1,
            "count": r.total_count,
            "diameters_mm": r.diameters_mm,
            "thread_specs": r.thread_specs,
            "slot_count": r.slot_count,
            "confidence": r.confidence,
            "source": r.source,
            "views": r.per_view,
            "notes": r.notes,
        })

    return {
        "total_unique_holes": total,
        "diameters_mm": sorted(all_diams),
        "thread_specs": all_threads,
        "total_slots": total_slots,
        "per_page": per_page,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup(lst: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in lst:
        key = item.lower().replace(" ", "")
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def _load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        logger.warning("Config file '%s' not found, using defaults", config_path)
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="OCRAI-FertigungKalk – extract manufacturing data from technical drawings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py drawing.pdf
  python main.py scan.tiff --output results/my_drawing.json
  python main.py drawing.pdf --config custom.json --no-debug
        """,
    )
    parser.add_argument("input_file", help="Path to PDF or TIFF drawing")
    parser.add_argument("--config", default="config.json",
                        help="Path to config JSON (default: config.json)")
    parser.add_argument("--output", default=None,
                        help="Path to write result JSON (default: <input_file>.result.json)")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable debug image generation")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity (default: INFO)")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config(args.config)

    if args.no_debug:
        cfg.setdefault("debug", {})["enabled"] = False

    try:
        result = run(args.input_file, cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 1
    except ValueError as exc:
        logger.error("Input error: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 2

    out_path = args.output or str(Path(args.input_file).with_suffix(".result.json"))
    output.write_json(result, out_path)
    output.print_summary(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
