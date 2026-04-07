"""
hole_detector.py
----------------
LLM vision-guided hole detection pipeline.

Two-stage approach:
  Stage 1 – Drawing Analysis (Claude Vision):
    Send the full rendered drawing image to Claude.
    Claude identifies:
      (a) all views / positions in the drawing (front, top, side, isometric, …)
      (b) how holes are represented in THIS drawing (circle + center mark, Ø annotation, M-thread, …)
      (c) preliminary hole specs per view

  Stage 2 – Hole Count (Claude Vision, per primary view):
    For each primary/main view, send a cropped sub-image.
    Claude counts all unique holes, threaded holes, and slots in that view.
    Isometric and auxiliary views are excluded to avoid double-counting.

  OCR Cross-Check:
    Verify LLM count against regex-extracted annotations ("Ø12", "M16", "2x Ø8.5").
    If both agree → high confidence.
    If they disagree → use LLM count (more reliable for visual features).

Returns HoleDetectionResult:
  total_count    : int
  diameters_mm   : list[float]  (unique diameters, sorted)
  thread_specs   : list[str]    e.g. ["M16", "M8x1.25"]
  slot_count     : int          oblong slots (Langlöcher)
  per_view       : list[dict]   breakdown per identified view
  confidence     : float        0.0 – 1.0
  source         : "llm_vision" | "ocr_only" | "combined"
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from input_loader import PageData
from ocr import OcrResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HoleDetectionResult:
    total_count: int = 0
    diameters_mm: list[float] = field(default_factory=list)
    thread_specs: list[str] = field(default_factory=list)
    slot_count: int = 0
    per_view: list[dict] = field(default_factory=list)
    confidence: float = 0.0
    source: str = "ocr_only"
    notes: str = ""


# ---------------------------------------------------------------------------
# Stage 1 prompt – drawing analysis
# ---------------------------------------------------------------------------

_STAGE1_SYSTEM = """\
You are an expert in reading technical engineering and manufacturing drawings
(DIN, ISO, ANSI standards). You can identify drawing views, hole symbols,
thread annotations, and manufacturing features.
Always respond with valid JSON only – no markdown, no explanation text."""

_STAGE1_PROMPT = """\
Analyze this technical drawing image carefully.

Your tasks:
1. Identify all VIEWS / POSITIONS shown (front view, top view, side view,
   isometric, section cut, detail view, etc.)
2. Describe how DRILL HOLES and THREADED HOLES are annotated in this specific
   drawing (e.g. circles with center marks, Ø prefix, M-thread notation,
   cross-hatch pattern, hidden lines, etc.)
3. For each view, count the holes visible in that view
4. Determine the TOTAL UNIQUE hole count
   (a hole shown in front AND top view still counts as ONE hole)
5. List all slot features (Langlöcher / oblong holes) separately

Return this exact JSON structure:
{
  "views": [
    {
      "name": "front_view",
      "type": "front|top|side|isometric|section|detail|auxiliary|unknown",
      "is_primary": true,
      "description": "short description",
      "hole_count_in_view": 2,
      "hole_annotations_found": ["Ø18", "2x Ø8.5"],
      "slot_count_in_view": 1
    }
  ],
  "hole_annotation_style": "describe how holes look in this drawing",
  "total_unique_holes": 2,
  "hole_diameters_mm": [8.5, 18.0],
  "thread_specs": ["M16"],
  "total_slots": 1,
  "slot_specs": ["18 wide"],
  "confidence": 0.9,
  "notes": "isometric view not counted to avoid double-counting"
}

RULES:
- Count each UNIQUE HOLE once (not once per view it appears in)
- Oblong slots / Langlöcher are NOT standard holes → put in slot_count
- R10, R5 etc. are corner radii, NOT holes → do not count
- Chamfers, fillets, boss features → NOT holes
- "NxØD" or "N×ØD" means N holes of diameter D → hole_count += N
- "ØD" alone → 1 hole of diameter D
- "MD" (e.g. M16) → threaded hole of nominal diameter D mm
- Confidence: 0.9+ if clear annotations, 0.5 if estimated visually only
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_holes(
    page: PageData,
    ocr: OcrResult,
    cfg: dict,
) -> HoleDetectionResult:
    """
    Run the full hole detection pipeline for one page.

    Falls back to OCR-only if the Anthropic API key is not set.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set – using OCR-only hole detection")
        return _ocr_only_fallback(ocr)

    llm_cfg = cfg.get("llm", {})

    # ── Stage 1: Analyze full drawing ────────────────────────────────────────
    logger.info("Hole detection Stage 1: LLM vision analysis …")
    stage1 = _llm_vision_call(
        image=page.image,
        prompt=_STAGE1_PROMPT,
        system=_STAGE1_SYSTEM,
        llm_cfg=llm_cfg,
        cache_prefix="holes_s1",
    )

    if stage1 is None:
        logger.warning("Stage 1 LLM call failed – falling back to OCR")
        return _ocr_only_fallback(ocr)

    result = _build_result_from_stage1(stage1)

    # ── OCR cross-check ───────────────────────────────────────────────────────
    ocr_count = _count_holes_from_ocr(ocr)
    result = _apply_ocr_crosscheck(result, ocr_count, ocr)

    logger.info(
        "Hole detection complete: count=%d, diameters=%s, slots=%d, confidence=%.2f, source=%s",
        result.total_count, result.diameters_mm, result.slot_count,
        result.confidence, result.source,
    )
    return result


# ---------------------------------------------------------------------------
# LLM vision call
# ---------------------------------------------------------------------------

def _llm_vision_call(
    image: np.ndarray,
    prompt: str,
    system: str,
    llm_cfg: dict,
    cache_prefix: str = "vision",
) -> Optional[dict]:
    """Encode image, call Claude vision API, return parsed JSON dict."""
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError("anthropic package required: pip install anthropic") from exc

    # Encode image as JPEG base64
    image_b64, media_type = _encode_image(image)

    # Cache key: hash of image + prompt
    cache_key = hashlib.sha256((image_b64[:200] + prompt).encode()).hexdigest()[:16]
    cached = _cache_load(f"{cache_prefix}_{cache_key}", llm_cfg)
    if cached is not None:
        logger.debug("LLM vision: cache hit %s", cache_key)
        return cached

    model = llm_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = int(llm_cfg.get("max_tokens_vision", 2048))

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = message.content[0].text.strip()
        logger.debug("LLM vision raw response (%d chars): %s …", len(raw), raw[:150])

        data = _parse_json(raw)
        if data:
            _cache_save(f"{cache_prefix}_{cache_key}", data, llm_cfg)
        return data

    except Exception as exc:
        logger.error("LLM vision call failed: %s", exc)
        return None


def _encode_image(image: np.ndarray) -> tuple[str, str]:
    """Encode BGR numpy image as JPEG base64. Returns (b64_string, media_type)."""
    # Downscale to max 1600px on longest side for API efficiency
    h, w = image.shape[:2]
    max_side = 1600
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return b64, "image/jpeg"


# ---------------------------------------------------------------------------
# Result building
# ---------------------------------------------------------------------------

def _build_result_from_stage1(data: dict) -> HoleDetectionResult:
    views = data.get("views", [])
    total = _int_or(data.get("total_unique_holes"), 0)
    diameters = _float_list(data.get("hole_diameters_mm"))
    threads = [str(t) for t in data.get("thread_specs", []) if t]
    slots = _int_or(data.get("total_slots"), 0)
    confidence = float(data.get("confidence", 0.7))
    notes = str(data.get("notes", ""))

    return HoleDetectionResult(
        total_count=total,
        diameters_mm=sorted(diameters),
        thread_specs=threads,
        slot_count=slots,
        per_view=views,
        confidence=confidence,
        source="llm_vision",
        notes=notes,
    )


# ---------------------------------------------------------------------------
# OCR cross-check
# ---------------------------------------------------------------------------

def _count_holes_from_ocr(ocr: OcrResult) -> int:
    """Estimate hole count from parsed OCR annotations."""
    import re
    total = 0
    # Pattern: optional multiplier × diameter spec
    _HOLE_COUNT_RE = re.compile(
        r"(\d+)\s*[xX×]\s*(?:[ØøΦ∅]|M)\s*\d",
        re.IGNORECASE,
    )
    for ann in ocr.hole_annotations:
        m = _HOLE_COUNT_RE.search(ann)
        if m:
            total += int(m.group(1))
        else:
            total += 1  # single hole

    return total


def _apply_ocr_crosscheck(
    result: HoleDetectionResult,
    ocr_count: int,
    ocr: OcrResult,
) -> HoleDetectionResult:
    """Merge LLM result with OCR data; flag discrepancies."""
    if ocr_count == 0:
        # OCR found nothing – trust LLM entirely
        return result

    if result.total_count == 0 and ocr_count > 0:
        # LLM missed holes that OCR found
        logger.info("OCR found %d holes that LLM missed – using OCR count", ocr_count)
        result.total_count = ocr_count
        result.source = "combined"
        result.confidence = min(result.confidence, 0.6)
        return result

    diff_pct = abs(result.total_count - ocr_count) / max(result.total_count, 1)
    if diff_pct > 0.3:
        logger.warning(
            "LLM count (%d) and OCR count (%d) disagree by %.0f%% – using LLM",
            result.total_count, ocr_count, diff_pct * 100,
        )
        result.confidence *= 0.85
        result.notes += f" [OCR cross-check discrepancy: OCR={ocr_count}, LLM={result.total_count}]"
    else:
        result.source = "combined"
        result.confidence = min(result.confidence + 0.05, 1.0)

    # Supplement diameters from OCR if LLM found none
    if not result.diameters_mm and ocr.hole_annotations:
        result.diameters_mm = _diameters_from_ocr(ocr)

    return result


def _diameters_from_ocr(ocr: OcrResult) -> list[float]:
    import re
    diams: list[float] = []
    _D_RE = re.compile(r"[ØøΦ∅]\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
    _M_RE = re.compile(r"M\s*(\d+(?:[.,]\d+)?)", re.IGNORECASE)
    for ann in ocr.hole_annotations:
        for m in _D_RE.finditer(ann):
            diams.append(float(m.group(1).replace(",", ".")))
        for m in _M_RE.finditer(ann):
            diams.append(float(m.group(1).replace(",", ".")))
    unique: list[float] = []
    for d in sorted(diams):
        if not any(abs(d - u) < 0.5 for u in unique):
            unique.append(d)
    return unique


# ---------------------------------------------------------------------------
# OCR-only fallback (no API key)
# ---------------------------------------------------------------------------

def _ocr_only_fallback(ocr: OcrResult) -> HoleDetectionResult:
    """Simple fallback using only OCR annotations when LLM is unavailable."""
    count = _count_holes_from_ocr(ocr)
    diams = _diameters_from_ocr(ocr)
    return HoleDetectionResult(
        total_count=count,
        diameters_mm=diams,
        confidence=0.5 if count > 0 else 0.0,
        source="ocr_only",
        notes="Anthropic API key not set; OCR-only detection",
    )


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _cache_path(key: str, llm_cfg: dict) -> Path:
    d = Path(llm_cfg.get("cache_dir", ".llm_cache"))
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{key}.json"


def _cache_load(key: str, llm_cfg: dict) -> Optional[dict]:
    if not llm_cfg.get("cache_enabled", True):
        return None
    p = _cache_path(key, llm_cfg)
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_save(key: str, data: dict, llm_cfg: dict) -> None:
    if not llm_cfg.get("cache_enabled", True):
        return
    try:
        _cache_path(key, llm_cfg).write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as exc:
        logger.debug("Cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def _parse_json(text: str) -> Optional[dict]:
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            if part.startswith("json"):
                part = part[4:]
            text = part.strip()
            break
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON: %s – text: %r", exc, text[:200])
        return None


# ---------------------------------------------------------------------------
# Type helpers
# ---------------------------------------------------------------------------

def _int_or(v, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float_list(v) -> list[float]:
    if not isinstance(v, list):
        return []
    result = []
    for item in v:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            pass
    return result
