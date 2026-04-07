"""
main.py
-------
Entry point for the OCRAI-FertigungKalk manufacturing data extraction system.

Pipeline:
  input → load → preprocess → geometry → OCR → scaling → LLM
       → classification → calculation → output

Usage:
  python main.py <input_file> [--config config.json] [--output result.json] [--no-debug]

  python main.py drawing.pdf
  python main.py scan.tiff --output results/my_drawing.json
  python main.py drawing.pdf --no-debug
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Module imports (all project modules)
# ---------------------------------------------------------------------------
import input_loader
import preprocessing
import geometry
import ocr as ocr_module
import scaling
import llm_interface
import classification
import calculation
import output
import debug as debug_module

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(input_file: str, cfg: dict) -> dict:
    """
    Execute the full extraction pipeline for one file.

    Returns the final output dict.
    """
    logger.info("=" * 60)
    logger.info("Processing: %s", input_file)
    logger.info("=" * 60)

    # ── 1. Load ─────────────────────────────────────────────────────────────
    pages = input_loader.load(input_file, cfg)
    logger.info("Loaded %d page(s)", len(pages))

    # Per-page accumulators
    page_calcs: list[calculation.PageCalculation] = []
    weld_per_page: list[classification.WeldClassification] = []
    holes_per_page: list[classification.HoleClassification] = []

    # Aggregate annotations across all pages for a single LLM call
    all_hole_annotations: list[str] = []
    all_weld_annotations: list[str] = []

    # ── 2–6. Per-page processing ─────────────────────────────────────────────
    page_intermediates = []

    for page in pages:
        logger.info("── Page %d / %d ──────────────────────────────────────", page.page_index + 1, len(pages))

        # 2. Preprocess
        pre = preprocessing.preprocess(page.image, cfg)

        # 3. Geometry extraction
        lines, circles = geometry.extract_geometry(page, pre, cfg)

        # 4. OCR / text extraction
        ocr_result = ocr_module.extract_text(page, pre, cfg)

        # 5. Scale detection
        scale_mm_per_px = scaling.compute_scale(
            ocr_result, lines, pre.edges, cfg,
            source_type=page.source_type,
            pdf_render_dpi=cfg.get("pdf_render_dpi", 300),
        )

        # Collect annotations for LLM
        all_hole_annotations.extend(ocr_result.hole_annotations)
        all_weld_annotations.extend(ocr_result.weld_annotations)

        page_intermediates.append((page, pre, lines, circles, ocr_result, scale_mm_per_px))

    # ── 6. Single LLM call for all collected annotations ────────────────────
    logger.info("Calling LLM for annotation interpretation …")
    llm_result = llm_interface.interpret_annotations(
        _dedup(all_hole_annotations),
        _dedup(all_weld_annotations),
        cfg,
    )
    logger.debug("LLM result: %s", llm_result)

    # ── 7–8. Classification + calculation per page ──────────────────────────
    for page, pre, lines, circles, ocr_result, scale_mm_per_px in page_intermediates:

        # 7a. Classify welds
        welds = classification.classify_welds(
            lines, ocr_result, llm_result, scale_mm_per_px, cfg
        )

        # 7b. Classify holes
        holes = classification.classify_holes(
            circles, ocr_result, llm_result, scale_mm_per_px, cfg
        )

        # 8. Calculate per-page times
        page_calc = calculation.calculate(welds, holes, page.page_index, cfg)

        weld_per_page.append(welds)
        holes_per_page.append(holes)
        page_calcs.append(page_calc)

        # 10. Debug visualisation
        debug_module.save_debug_page(
            page.page_index, page.image, pre,
            welds, holes, ocr_result,
            scale_mm_per_px, cfg,
        )

    # ── 9. Aggregate + format output ─────────────────────────────────────────
    calc_result = calculation.aggregate(page_calcs, cfg)
    result = output.build_output(calc_result, weld_per_page, holes_per_page, input_file)

    return result


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
    parser.add_argument(
        "--config", default="config.json",
        help="Path to config JSON (default: config.json)",
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to write result JSON (default: <input_file>.result.json)",
    )
    parser.add_argument(
        "--no-debug", action="store_true",
        help="Disable debug image generation",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO)",
    )
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
        if "debug" not in cfg:
            cfg["debug"] = {}
        cfg["debug"]["enabled"] = False

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

    # Determine output path
    out_path = args.output
    if out_path is None:
        out_path = str(Path(args.input_file).with_suffix(".result.json"))

    output.write_json(result, out_path)
    output.print_summary(result)

    return 0


if __name__ == "__main__":
    sys.exit(main())
