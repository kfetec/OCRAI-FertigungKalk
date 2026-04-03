"""
llm_interface.py
----------------
LLM (Claude via Anthropic API) integration.

ONLY used for:
  - Parsing ambiguous or compound OCR annotation strings
  - Converting raw text snippets into structured manufacturing data

Results are cached to disk to avoid redundant API calls.

Output schema (always a dict):
{
  "hole_count"   : int | null,
  "hole_diameters": [float, …] | null,
  "weld_type"    : str | null,       e.g. "fillet", "butt"
  "weld_size"    : float | null,     e.g. 5  (throat size in mm)
  "weld_length"  : float | null,     if explicitly annotated
  "is_all_around": bool,             umlaufend / allround
  "extra"        : dict              any other parsed fields
}
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a specialist in reading German and English technical drawing annotations
for steel construction / structural steel fabrication.

Your task: parse the given annotation strings and return a JSON object.

Respond ONLY with valid JSON – no markdown, no explanation.

JSON schema:
{
  "hole_count":    <integer or null>,
  "hole_diameters": [<float>, ...] or null,
  "weld_type":     "<fillet|butt|plug|slot|spot|seam|other>" or null,
  "weld_size":     <float or null>,
  "weld_length":   <float or null>,
  "is_all_around": <true|false>,
  "extra":         {}
}

Rules:
- "umlaufend" or "allround" means is_all_around = true
- "Kehlnaht" / "Kehlnähte" → weld_type = "fillet"
- "Stumpfnaht" → weld_type = "butt"
- "a=N" means weld_size = N (throat depth in mm)
- "ØN" or "∅N" means hole diameter N mm
- "MN" means threaded hole diameter N mm → include in hole_diameters
- "NxØM" means hole_count = N, hole_diameters = [M]
- If multiple diameters appear, list all of them
- All numeric values must be numbers, not strings
"""


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpret_annotations(
    hole_annotations: list[str],
    weld_annotations: list[str],
    cfg: dict,
) -> dict:
    """
    Send annotations to Claude and return structured manufacturing data.

    Parameters
    ----------
    hole_annotations : list[str]   from ocr.OcrResult
    weld_annotations : list[str]   from ocr.OcrResult
    cfg              : dict        top-level config

    Returns
    -------
    dict  matching schema described above
    """
    if not hole_annotations and not weld_annotations:
        logger.debug("LLM: no annotations to interpret, returning empty result")
        return _empty_result()

    llm_cfg = cfg.get("llm", {})
    prompt_text = _build_prompt(hole_annotations, weld_annotations)

    # Check cache first
    cache_key = hashlib.sha256(prompt_text.encode()).hexdigest()[:16]
    cached = _cache_load(cache_key, llm_cfg)
    if cached is not None:
        logger.debug("LLM: cache hit for key %s", cache_key)
        return cached

    result = _call_claude(prompt_text, llm_cfg)
    _cache_save(cache_key, result, llm_cfg)
    return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_prompt(hole_annotations: list[str], weld_annotations: list[str]) -> str:
    lines = ["The following annotations were extracted from a technical drawing:"]
    lines.append("")
    for ann in hole_annotations:
        lines.append(f'  - "{ann}"')
    for ann in weld_annotations:
        lines.append(f'  - "{ann}"')
    lines.append("")
    lines.append("Parse these annotations and return the JSON object.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def _call_claude(prompt: str, llm_cfg: dict) -> dict:
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "anthropic package required. Install via: pip install anthropic"
        ) from exc

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        logger.warning("ANTHROPIC_API_KEY not set – LLM interpretation skipped")
        return _empty_result()

    model = llm_cfg.get("model", "claude-sonnet-4-6")
    max_tokens = int(llm_cfg.get("max_tokens", 1024))
    temperature = float(llm_cfg.get("temperature", 0))

    client = anthropic.Anthropic(api_key=api_key)

    try:
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        raw_text = message.content[0].text.strip()
        logger.debug("LLM raw response: %s", raw_text[:200])
        return _parse_llm_response(raw_text)

    except anthropic.APIError as exc:
        logger.error("Anthropic API error: %s", exc)
        return _empty_result()
    except Exception as exc:
        logger.error("Unexpected LLM error: %s", exc)
        return _empty_result()


def _parse_llm_response(text: str) -> dict:
    """Parse JSON from LLM response, with fallback for markdown fences."""
    # Strip potential markdown code fence
    if "```" in text:
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning("LLM returned invalid JSON (%s): %r", exc, text[:200])
        return _empty_result()

    # Normalise / validate fields
    result = _empty_result()
    result["hole_count"] = _int_or_none(data.get("hole_count"))
    result["hole_diameters"] = _float_list_or_none(data.get("hole_diameters"))
    result["weld_type"] = str(data["weld_type"]) if data.get("weld_type") else None
    result["weld_size"] = _float_or_none(data.get("weld_size"))
    result["weld_length"] = _float_or_none(data.get("weld_length"))
    result["is_all_around"] = bool(data.get("is_all_around", False))
    result["extra"] = dict(data.get("extra", {}))
    return result


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

def _cache_path(cache_key: str, llm_cfg: dict) -> Path:
    cache_dir = Path(llm_cfg.get("cache_dir", ".llm_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{cache_key}.json"


def _cache_load(cache_key: str, llm_cfg: dict) -> Optional[dict]:
    if not llm_cfg.get("cache_enabled", True):
        return None
    path = _cache_path(cache_key, llm_cfg)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None


def _cache_save(cache_key: str, data: dict, llm_cfg: dict) -> None:
    if not llm_cfg.get("cache_enabled", True):
        return
    path = _cache_path(cache_key, llm_cfg)
    try:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Cache write failed: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _empty_result() -> dict:
    return {
        "hole_count": None,
        "hole_diameters": None,
        "weld_type": None,
        "weld_size": None,
        "weld_length": None,
        "is_all_around": False,
        "extra": {},
    }


def _int_or_none(v) -> Optional[int]:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _float_or_none(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _float_list_or_none(v) -> Optional[list[float]]:
    if v is None:
        return None
    if not isinstance(v, list):
        return None
    result = []
    for item in v:
        try:
            result.append(float(item))
        except (TypeError, ValueError):
            pass
    return result if result else None
