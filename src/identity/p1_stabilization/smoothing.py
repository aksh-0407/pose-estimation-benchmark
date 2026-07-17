"""One-Euro filtering of per-keypoint pixel trajectories.

The One-Euro filter (Casiez, Roussel & Vogel, CHI 2012  - 
https://gery.casiez.net/1euro/) is the standard low-latency jitter filter for noisy
interactive signals: a low-pass filter whose cutoff rises with the signal speed, so it
removes jitter when a joint is still without adding lag when it moves fast. We run one
filter per keypoint coordinate along a micro-track, with a confidence-gated spike clamp
in front of it so a single hallucinated keypoint cannot drag the trajectory.
"""

from __future__ import annotations

import math

import numpy as np

from identity.p1_stabilization.config import GatingConfig, SmoothingConfig


def _alpha(cutoff: float, dt: float) -> float:
    tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
    return 1.0 / (1.0 + tau / max(dt, 1e-6))


class OneEuroFilter:
    """Scalar One-Euro filter."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0

    def filter(self, value: float, dt: float) -> float:
        if self._x_prev is None or not math.isfinite(self._x_prev):
            self._x_prev = value
            self._dx_prev = 0.0
            return value
        dx = (value - self._x_prev) / max(dt, 1e-6)
        dx_hat = self._dx_prev + _alpha(self.d_cutoff, dt) * (dx - self._dx_prev)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        x_hat = self._x_prev + _alpha(cutoff, dt) * (value - self._x_prev)
        self._x_prev = x_hat
        self._dx_prev = dx_hat
        return x_hat

    @property
    def last(self) -> float | None:
        return self._x_prev


def smooth_track_keypoints(
    keypoints_series: np.ndarray,   # (T, K, 2) pixel coords
    confidence_series: np.ndarray,  # (T, K)
    bbox_diag_series: np.ndarray,   # (T,) bbox diagonal length (px) per frame
    dt_series: np.ndarray,          # (T,) seconds since previous sample (dt[0] arbitrary)
    smoothing: SmoothingConfig,
    gating: GatingConfig,
) -> np.ndarray:
    """Return a (T, K, 2) smoothed copy of ``keypoints_series``.

    A keypoint that is invalid (non-finite) or exactly (0, 0) - the P1 placeholder for a
    missing joint - is passed through untouched and does not seed/advance its filter.
    """
    T, K, _ = keypoints_series.shape
    out = np.array(keypoints_series, dtype=float, copy=True)
    for k in range(K):
        fx = OneEuroFilter(smoothing.min_cutoff, smoothing.beta, smoothing.d_cutoff)
        fy = OneEuroFilter(smoothing.min_cutoff, smoothing.beta, smoothing.d_cutoff)
        for t in range(T):
            x, y = float(keypoints_series[t, k, 0]), float(keypoints_series[t, k, 1])
            conf = float(confidence_series[t, k])
            if not (math.isfinite(x) and math.isfinite(y)) or (x == 0.0 and y == 0.0):
                continue  # missing/placeholder joint - leave as-is, don't advance the filter
            # Confidence-gated spike clamp: a big jump on a low-confidence keypoint is
            # replaced by the last filtered position before it can pollute the filter.
            last_x, last_y = fx.last, fy.last
            if last_x is not None and last_y is not None and conf < gating.confidence_min:
                jump = math.hypot(x - last_x, y - last_y)
                limit = max(gating.max_jump_px, gating.max_jump_bbox_frac * float(bbox_diag_series[t]))
                if jump > limit:
                    x, y = last_x, last_y
            dt = float(dt_series[t]) if math.isfinite(dt_series[t]) and dt_series[t] > 0 else 1e-3
            out[t, k, 0] = fx.filter(x, dt)
            out[t, k, 1] = fy.filter(y, dt)
    return out


def mean_jitter_px(keypoints_series: np.ndarray, confidence_series: np.ndarray,
                   confidence_min: float) -> float:
    """Mean frame-to-frame displacement (px) over confident, valid keypoints - the
    'how noisy is this trajectory' number used before/after to prove the smoothing."""
    T = keypoints_series.shape[0]
    if T < 2:
        return 0.0
    diffs: list[float] = []
    for t in range(1, T):
        prev, cur = keypoints_series[t - 1], keypoints_series[t]
        cp, cc = confidence_series[t - 1], confidence_series[t]
        for k in range(keypoints_series.shape[1]):
            if cp[k] < confidence_min or cc[k] < confidence_min:
                continue
            if (prev[k, 0] == 0.0 and prev[k, 1] == 0.0) or (cur[k, 0] == 0.0 and cur[k, 1] == 0.0):
                continue
            if not np.all(np.isfinite(prev[k])) or not np.all(np.isfinite(cur[k])):
                continue
            diffs.append(float(math.hypot(cur[k, 0] - prev[k, 0], cur[k, 1] - prev[k, 1])))
    return float(np.mean(diffs)) if diffs else 0.0
