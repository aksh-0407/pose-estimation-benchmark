"""Lightweight, illumination-tolerant player colour descriptors for P3.

The descriptor intentionally represents kit colour, not identity by itself.
Geometry remains the primary cue; appearance prevents obvious cross-team and
official/player mismatches when two ground locations are close.
"""

from __future__ import annotations

import cv2
import numpy as np


def extract_appearance_descriptor(image_bgr: np.ndarray, player: dict) -> np.ndarray | None:
    """Extract a RootSIFT-style HSV histogram from the central body region."""

    image = np.asarray(image_bgr)
    bbox = np.asarray(player.get("bbox_xywh_px", []), dtype=float)
    if image.ndim != 3 or image.shape[2] != 3 or bbox.shape != (4,):
        return None
    if not np.isfinite(bbox).all() or bbox[2] < 8.0 or bbox[3] < 16.0:
        return None
    height, width = image.shape[:2]
    x, y, w, h = bbox
    # Exclude head, hands, bat, and most background. The centre-body crop is
    # more stable than a pose polygon when shoulders/hips are partly occluded.
    x1 = int(np.clip(np.floor(x + 0.18 * w), 0, width))
    x2 = int(np.clip(np.ceil(x + 0.82 * w), 0, width))
    y1 = int(np.clip(np.floor(y + 0.12 * h), 0, height))
    y2 = int(np.clip(np.ceil(y + 0.72 * h), 0, height))
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    roi = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # Ignore nearly black and near-grey pixels; these are dominated by shadows,
    # pitch/background, white skeleton overlays, and exposure differences.
    mask = ((hsv[..., 1] >= 24) & (hsv[..., 2] >= 24)).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < 24:
        mask = (hsv[..., 2] >= 20).astype(np.uint8) * 255
    if int(np.count_nonzero(mask)) < 24:
        return None
    histogram = cv2.calcHist([hsv], [0, 1], mask, [24, 8], [0, 180, 0, 256]).reshape(-1)
    total = float(histogram.sum())
    if total <= 0.0 or not np.isfinite(histogram).all():
        return None
    # Square-rooted L1 histogram makes Euclidean distance equal to Hellinger.
    return np.sqrt(histogram / total).astype(np.float32)


def appearance_distance(left: np.ndarray | None, right: np.ndarray | None) -> float | None:
    """Hellinger distance in ``[0, 1]`` or ``None`` when either cue is absent."""

    if left is None or right is None:
        return None
    a, b = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return None
    return float(np.clip(np.linalg.norm(a - b) / np.sqrt(2.0), 0.0, 1.0))
