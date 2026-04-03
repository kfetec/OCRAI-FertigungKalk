"""
preprocessing.py
----------------
Image preprocessing for raster inputs (PDF fallback + TIFF).

Pipeline:
  1. Grayscale conversion
  2. Optional deskewing
  3. Gaussian blur (noise reduction)
  4. Canny edge detection
  5. Adaptive thresholding (optional, returned separately)

Returns a PreprocessResult with all intermediate images for downstream use.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    gray: np.ndarray          # Grayscale original
    blurred: np.ndarray       # After Gaussian blur
    edges: np.ndarray         # Canny edge map
    thresh: np.ndarray        # Adaptive threshold binary
    deskew_angle_deg: float   # Detected skew angle (0 if not applied)
    scale_factor: float       # Downscale factor used (1.0 = full res)
    # Downscaled variants for fast detection
    gray_small: np.ndarray
    edges_small: np.ndarray


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess(image: np.ndarray, cfg: dict) -> PreprocessResult:
    """
    Run the full preprocessing pipeline on a BGR image.

    Parameters
    ----------
    image : np.ndarray  BGR image (full resolution)
    cfg   : dict        Top-level config (reads cfg["preprocessing"] sub-key)

    Returns
    -------
    PreprocessResult
    """
    pcfg = cfg.get("preprocessing", {})
    downscale = cfg.get("image_downscale_factor", 0.5)

    gray = _to_gray(image)

    # Deskew before any other processing
    deskew_angle = 0.0
    try:
        gray, deskew_angle = _deskew(gray)
    except Exception as exc:
        logger.debug("Deskew skipped: %s", exc)

    # Noise reduction
    ksize = int(pcfg.get("gaussian_blur_kernel", 3))
    if ksize % 2 == 0:
        ksize += 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)

    # Edge detection
    lo = int(pcfg.get("canny_threshold_low", 50))
    hi = int(pcfg.get("canny_threshold_high", 150))
    edges = cv2.Canny(blurred, lo, hi)

    # Adaptive threshold for OCR pre-processing
    block = int(pcfg.get("adaptive_block_size", 15))
    if block % 2 == 0:
        block += 1
    c_val = int(pcfg.get("adaptive_c", 10))
    thresh = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block, c_val,
    )

    # Downscaled versions for fast Hough detection
    scale_factor = float(downscale)
    if abs(scale_factor - 1.0) > 0.01:
        new_w = max(1, int(gray.shape[1] * scale_factor))
        new_h = max(1, int(gray.shape[0] * scale_factor))
        gray_small = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
        edges_small = cv2.resize(edges, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        gray_small = gray
        edges_small = edges
        scale_factor = 1.0

    logger.debug(
        "Preprocessing done: shape=%s, skew=%.2f°, scale=%.2f",
        gray.shape, deskew_angle, scale_factor,
    )

    return PreprocessResult(
        gray=gray,
        blurred=blurred,
        edges=edges,
        thresh=thresh,
        deskew_angle_deg=deskew_angle,
        scale_factor=scale_factor,
        gray_small=gray_small,
        edges_small=edges_small,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3:
        if image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unexpected image shape: {image.shape}")


def _deskew(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Estimate and correct scan skew using projection-profile heuristic.
    Returns (corrected_image, angle_deg).
    """
    # Use Canny + HoughLines on a downscaled version for speed
    small = cv2.resize(gray, (0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    edges = cv2.Canny(small, 30, 100)
    lines = cv2.HoughLines(edges, 1, math.pi / 180, threshold=60)

    if lines is None or len(lines) == 0:
        return gray, 0.0

    angles = []
    for line in lines[:50]:
        theta = line[0][1]
        angle_deg = math.degrees(theta) - 90.0
        # Only consider near-horizontal lines (within ±10°)
        if abs(angle_deg) <= 10.0:
            angles.append(angle_deg)

    if not angles:
        return gray, 0.0

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.3:  # negligible skew
        return gray, median_angle

    h, w = gray.shape
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), median_angle, 1.0)
    corrected = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    logger.debug("Deskew applied: %.2f°", median_angle)
    return corrected, median_angle


def enhance_for_ocr(image: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Enhance a grayscale or BGR image for better OCR accuracy.
    Returns a grayscale uint8 image.
    """
    ocr_cfg = cfg.get("ocr", {})
    alpha = float(ocr_cfg.get("contrast_alpha", 1.5))
    beta = int(ocr_cfg.get("contrast_beta", 20))

    gray = _to_gray(image)
    enhanced = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    # Binarize with Otsu
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
