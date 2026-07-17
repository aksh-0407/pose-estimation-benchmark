"""Constant-velocity Kalman filter for bbox tracking (centre + size state)."""

from __future__ import annotations

import numpy as np

# State: [cx, cy, w, h, vcx, vcy, vw, vh]; measurement: [cx, cy, w, h]
_NDIM = 4
_MIN_SIZE_PX = 1.0


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


class KalmanBoxTracker:
    def __init__(self, bbox_xywh: list[float]) -> None:
        cx, cy, w, h = self._to_cxcywh(bbox_xywh)
        self._x = np.array([cx, cy, w, h, 0.0, 0.0, 0.0, 0.0], dtype=float)

        self._F = np.eye(8)
        for i in range(_NDIM):
            self._F[i, i + _NDIM] = 1.0  # x += v each step

        self._H = np.zeros((_NDIM, 8))
        self._H[:_NDIM, :_NDIM] = np.eye(_NDIM)

        self._P = np.eye(8) * 10.0
        self._P[4:, 4:] *= 1000.0  # high initial velocity uncertainty
        self._q = 1.0   # process-noise scale (inflated while dormant)
        self._r = 1.0   # measurement-noise scale

    @staticmethod
    def _to_cxcywh(bbox_xywh: list[float]) -> tuple[float, float, float, float]:
        x, y, w, h = [float(v) for v in bbox_xywh]
        w = max(w, _MIN_SIZE_PX)
        h = max(h, _MIN_SIZE_PX)
        return x + w / 2.0, y + h / 2.0, w, h

    def _Q(self) -> np.ndarray:
        q = np.eye(8)
        q[:_NDIM, :_NDIM] *= self._q
        q[_NDIM:, _NDIM:] *= self._q * 0.01
        return q

    def _R(self) -> np.ndarray:
        return np.eye(_NDIM) * self._r

    def predict(self) -> None:
        self._x = self._F @ self._x
        self._x[2:4] = np.maximum(self._x[2:4], _MIN_SIZE_PX)
        self._P = _symmetrize(self._F @ self._P @ self._F.T + self._Q())

    def update(self, bbox_xywh: list[float]) -> None:
        # A re-acquired track is being measured again - restore nominal
        # process noise (mark_missed inflates it 1.5x per missed frame; without
        # this reset a 10-frame gap left _q ~ 57 for the track's whole life).
        self._q = 1.0
        z = np.array(self._to_cxcywh(bbox_xywh), dtype=float)
        S = self._H @ self._P @ self._H.T + self._R()
        K = np.linalg.solve(S, self._H @ self._P).T
        self._x = self._x + K @ (z - self._H @ self._x)
        self._x[2:4] = np.maximum(self._x[2:4], _MIN_SIZE_PX)
        identity = np.eye(8)
        innovation = identity - K @ self._H
        self._P = _symmetrize(innovation @ self._P @ innovation.T + K @ self._R() @ K.T)

    def predicted_bbox(self) -> np.ndarray:
        cx, cy, w, h = self._x[:_NDIM]
        w, h = max(float(w), _MIN_SIZE_PX), max(float(h), _MIN_SIZE_PX)
        return np.array([cx - w / 2.0, cy - h / 2.0, w, h])

    def center(self) -> np.ndarray:
        return self._x[:2].copy()

    def velocity(self) -> np.ndarray:
        return self._x[4:6].copy()

    def bbox_height(self) -> float:
        return max(float(self._x[3]), _MIN_SIZE_PX)

    def position_cov_trace(self) -> float:
        return float(np.trace(self._P[:2, :2]))

    def gating_distance_sq(self, center_xy: np.ndarray) -> float:
        S = self._P[:2, :2] + np.eye(2) * self._r
        diff = np.asarray(center_xy, dtype=float) - self._x[:2]
        if diff.shape != (2,) or not np.isfinite(diff).all() or not np.isfinite(S).all():
            return float("inf")
        try:
            solved = np.linalg.solve(S, diff)
        except np.linalg.LinAlgError:
            return float("inf")
        distance = float(diff @ solved)
        return distance if np.isfinite(distance) else float("inf")

    def inflate_process_noise(self, factor: float) -> None:
        self._q = min(self._q * float(factor), 1.0e5)

    def reupdate_virtual(self, prev_bbox_xywh: list[float], new_bbox_xywh: list[float], gap: int) -> None:
        """OC-SORT ORU: re-derive the state from a virtual straight trajectory.

        A track lost for ``gap`` frames has dead-reckoned on its stale pre-gap velocity
        (the constant-velocity error that fragments sharp manoeuvres). On recovery, re-anchor
        the filter at the last real observation and walk predict()/update() along ``gap``
        linearly-interpolated virtual observations to the new one, so the velocity and
        covariance reflect the ACTUAL displacement over the gap rather than the drift.
        Ends exactly at ``new_bbox_xywh`` (its final update), so the posterior matches a
        normal update but with a corrected trajectory behind it.
        """
        steps = max(int(gap), 1)
        p = np.array(self._to_cxcywh(prev_bbox_xywh), dtype=float)
        n = np.array(self._to_cxcywh(new_bbox_xywh), dtype=float)
        # Re-anchor at the last real observation with a fresh (uncertain) covariance.
        self._x = np.array([p[0], p[1], p[2], p[3], 0.0, 0.0, 0.0, 0.0], dtype=float)
        self._P = np.eye(8) * 10.0
        self._P[4:, 4:] *= 1000.0
        self._q = 1.0
        for s in range(1, steps + 1):
            v = p + (n - p) * (s / steps)
            self.predict()
            self.update([v[0] - v[2] / 2.0, v[1] - v[3] / 2.0, v[2], v[3]])

    def reseed(self, bbox_xywh: list[float], keep_velocity: np.ndarray) -> None:
        cx, cy, w, h = self._to_cxcywh(bbox_xywh)
        self._x = np.array(
            [cx, cy, w, h, float(keep_velocity[0]), float(keep_velocity[1]), 0.0, 0.0],
            dtype=float,
        )
        self._P = np.eye(8) * 10.0
        self._P[4:, 4:] *= 1000.0
        self._q = 1.0
