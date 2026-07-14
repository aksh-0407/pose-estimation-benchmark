"""Empirical cue calibration for cross-camera identity.

There is no identity ground truth, so calibration is bootstrapped from geometry:

* **same-player anchors** — cross-camera tracklet pairs whose ground projections
  agree tightly for many frames while no other player is nearby (unambiguous by
  isolation, e.g. the bowler's run-up or the square-leg umpire);
* **different-player pairs** — co-visible cross-camera tracklet pairs whose ground
  projections stay metres apart.

From those two populations every cue (ground residual, appearance distance,
posture z) gets robust same/different Gaussians, and cue values are scored as
log-likelihood ratios. A cue that cannot separate the populations on THIS footage
gets a d' near zero and its clipped LLR contributes almost nothing — cues degrade
to abstention instead of lying, which is the safety property the association
solver relies on.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from identity.common.pose_shape import DEFAULT_POSTURE_SIGMA_SYS

SCHEMA_VERSION = "cue_calibration/v1"

# Conservative defaults used before/without a fitted calibration file. Scales are
# chosen from the measured baseline run (cross-camera ground spread p95 ~2.5 m,
# Hellinger kit distances ~0.2 same-team) and err toward weak, low-confidence LLRs.
_DEFAULT_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    # per-frame euclidean ground residual between co-observing cameras (metres)
    "ground_dist_m": {"mu_same": 0.9, "sigma_same": 0.55, "mu_diff": 4.5, "sigma_diff": 2.5},
    # sqrt of covariance-normalized ground Mahalanobis^2
    "ground_maha": {"mu_same": 1.1, "sigma_same": 0.6, "mu_diff": 5.0, "sigma_diff": 2.5},
    # Hellinger appearance distance in [0, 1]
    "appearance": {"mu_same": 0.25, "sigma_same": 0.10, "mu_diff": 0.45, "sigma_diff": 0.15},
    # RMS posture z-score (dimensionless; ~1 for same player by construction)
    "posture_z": {"mu_same": 1.1, "sigma_same": 0.6, "mu_diff": 3.5, "sigma_diff": 1.8},
}

_MIN_SIGMA = 1e-3

# Per-cue sigma floors for fitting. Per-frame samples within one tracklet pair are
# heavily correlated, so a fit from few pairs can produce an absurdly narrow sigma
# (observed: 0.02 on appearance from a single anchor pair) that turns a weak cue
# into a hard veto. The floors keep every fitted distribution honest.
_SIGMA_FLOORS: dict[str, float] = {
    "ground_dist_m": 0.25,
    "ground_maha": 0.35,
    "appearance": 0.06,
    "posture_z": 0.5,
}


def _robust_gaussian(values: list[float] | np.ndarray) -> tuple[float, float] | None:
    data = np.asarray([v for v in np.asarray(values, dtype=float).ravel() if np.isfinite(v)])
    if data.size < 8:
        return None
    centre = float(np.median(data))
    sigma = 1.4826 * float(np.median(np.abs(data - centre)))
    if not np.isfinite(sigma) or sigma < _MIN_SIGMA:
        sigma = max(float(np.std(data)), _MIN_SIGMA)
    return centre, sigma


@dataclass(frozen=True)
class CueDistribution:
    mu_same: float
    sigma_same: float
    mu_diff: float
    sigma_diff: float
    n_same: int = 0
    n_diff: int = 0
    fitted: bool = False

    def d_prime(self) -> float:
        pooled = math.sqrt(0.5 * (self.sigma_same ** 2 + self.sigma_diff ** 2))
        if pooled < _MIN_SIGMA:
            return 0.0
        return abs(self.mu_diff - self.mu_same) / pooled

    def llr(self, value: float, *, clip: float = 4.0, clip_pos: float | None = None) -> float:
        """log p(value | same) - log p(value | different), asymmetrically clippable.

        ``clip_pos`` caps the positive side separately: agreement on a cue is weak
        evidence of identity (two players can share position, kit, and build) while
        strong disagreement is near-conclusive evidence of difference.
        """

        if not np.isfinite(value):
            return 0.0
        sigma_same = max(self.sigma_same, _MIN_SIGMA)
        sigma_diff = max(self.sigma_diff, _MIN_SIGMA)
        log_same = -0.5 * ((value - self.mu_same) / sigma_same) ** 2 - math.log(sigma_same)
        log_diff = -0.5 * ((value - self.mu_diff) / sigma_diff) ** 2 - math.log(sigma_diff)
        upper = clip if clip_pos is None else clip_pos
        return float(np.clip(log_same - log_diff, -clip, upper))

    def to_json(self) -> dict:
        return {
            "mu_same": self.mu_same, "sigma_same": self.sigma_same,
            "mu_diff": self.mu_diff, "sigma_diff": self.sigma_diff,
            "n_same": self.n_same, "n_diff": self.n_diff,
            "fitted": self.fitted, "d_prime": self.d_prime(),
        }

    @staticmethod
    def from_json(payload: dict) -> "CueDistribution":
        return CueDistribution(
            mu_same=float(payload["mu_same"]), sigma_same=float(payload["sigma_same"]),
            mu_diff=float(payload["mu_diff"]), sigma_diff=float(payload["sigma_diff"]),
            n_same=int(payload.get("n_same", 0)), n_diff=int(payload.get("n_diff", 0)),
            fitted=bool(payload.get("fitted", False)),
        )


def _default_distribution(cue: str) -> CueDistribution:
    values = _DEFAULT_DISTRIBUTIONS[cue]
    return CueDistribution(fitted=False, n_same=0, n_diff=0, **values)


@dataclass
class CueCalibration:
    """Fitted (or default) distributions per cue + posture systematic sigmas.

    Appearance is additionally calibrated PER CAMERA PAIR: colour processing can
    differ between cameras (measured on this rig: the pano camera shifts kit
    colours more than kits differ between people), so a global appearance model
    punishes exactly the pairs it was never fitted on. A camera pair without its
    own separable fit abstains.
    """

    distributions: dict[str, CueDistribution] = field(
        default_factory=lambda: {cue: _default_distribution(cue) for cue in _DEFAULT_DISTRIBUTIONS}
    )
    appearance_by_pair: dict[str, CueDistribution] = field(default_factory=dict)
    posture_sigma_sys: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_POSTURE_SIGMA_SYS)
    )
    anchor_pair_count: int = 0
    diff_pair_count: int = 0

    @staticmethod
    def camera_pair_key(cam_a: str, cam_b: str) -> str:
        return "|".join(sorted((cam_a, cam_b)))

    def appearance_llr(
        self, cam_a: str, cam_b: str, value: float | None, *,
        clip: float = 4.0, clip_pos: float | None = None, min_d_prime: float = 0.5,
    ) -> float:
        if value is None:
            return 0.0
        dist = self.appearance_by_pair.get(self.camera_pair_key(cam_a, cam_b))
        if dist is None or not dist.fitted or dist.d_prime() < min_d_prime:
            return 0.0
        return dist.llr(float(value), clip=clip, clip_pos=clip_pos)

    def llr(
        self, cue: str, value: float | None, *,
        clip: float = 4.0, clip_pos: float | None = None,
    ) -> float:
        if value is None:
            return 0.0
        dist = self.distributions.get(cue)
        if dist is None:
            return 0.0
        return dist.llr(float(value), clip=clip, clip_pos=clip_pos)

    def d_prime(self, cue: str) -> float:
        dist = self.distributions.get(cue)
        return dist.d_prime() if dist is not None else 0.0

    def summary(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "anchor_pair_count": self.anchor_pair_count,
            "diff_pair_count": self.diff_pair_count,
            "cues": {name: dist.to_json() for name, dist in sorted(self.distributions.items())},
            "appearance_by_pair": {
                key: dist.to_json() for key, dist in sorted(self.appearance_by_pair.items())
            },
            "posture_sigma_sys": {k: float(v) for k, v in sorted(self.posture_sigma_sys.items())},
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2, sort_keys=True)
            handle.write("\n")

    @staticmethod
    def load(path: str | Path) -> "CueCalibration":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        calibration = CueCalibration()
        for name, dist_payload in payload.get("cues", {}).items():
            calibration.distributions[name] = CueDistribution.from_json(dist_payload)
        for key, dist_payload in payload.get("appearance_by_pair", {}).items():
            calibration.appearance_by_pair[key] = CueDistribution.from_json(dist_payload)
        for name, value in payload.get("posture_sigma_sys", {}).items():
            if np.isfinite(value) and value > 0:
                calibration.posture_sigma_sys[name] = float(value)
        calibration.anchor_pair_count = int(payload.get("anchor_pair_count", 0))
        calibration.diff_pair_count = int(payload.get("diff_pair_count", 0))
        return calibration


def fit_cue_calibration(
    *,
    same_samples: dict[str, list[float]],
    diff_samples: dict[str, list[float]],
    posture_same_deltas: dict[str, list[float]] | None = None,
    anchor_pair_count: int = 0,
    diff_pair_count: int = 0,
) -> CueCalibration:
    """Fit per-cue same/different Gaussians; fall back to defaults per cue when thin.

    ``posture_same_deltas`` are raw |delta metres| per posture quantity across
    anchor pairs; their RMS becomes the cross-camera systematic sigma used to
    z-score posture aggregates.
    """

    calibration = CueCalibration(
        anchor_pair_count=anchor_pair_count, diff_pair_count=diff_pair_count
    )
    for cue in sorted(set(same_samples) | set(diff_samples) | set(_DEFAULT_DISTRIBUTIONS)):
        same_fit = _robust_gaussian(same_samples.get(cue, []))
        diff_fit = _robust_gaussian(diff_samples.get(cue, []))
        if same_fit is None or diff_fit is None:
            continue  # keep the conservative default for this cue
        floor = _SIGMA_FLOORS.get(cue, _MIN_SIGMA)
        mu_same, sigma_same = same_fit[0], max(same_fit[1], floor)
        mu_diff, sigma_diff = diff_fit[0], max(diff_fit[1], floor)
        if mu_diff <= mu_same:  # cue carries no usable direction on this footage
            # Collapse to a near-zero-information distribution rather than an
            # inverted one: identical means, pooled sigma.
            pooled = max(sigma_same, sigma_diff, _MIN_SIGMA)
            mu_diff = mu_same
            sigma_same = sigma_diff = pooled
        calibration.distributions[cue] = CueDistribution(
            mu_same=mu_same, sigma_same=sigma_same,
            mu_diff=mu_diff, sigma_diff=sigma_diff,
            n_same=len(same_samples.get(cue, [])),
            n_diff=len(diff_samples.get(cue, [])),
            fitted=True,
        )
    if posture_same_deltas:
        for name, deltas in posture_same_deltas.items():
            data = np.asarray([d for d in deltas if np.isfinite(d)], dtype=float)
            if data.size >= 6:
                rms = float(np.sqrt(np.mean(np.square(data))))
                if np.isfinite(rms) and rms > 0.005:
                    calibration.posture_sigma_sys[name] = rms
    return calibration


def fit_pair_distribution(
    same_values: list[float],
    diff_values: list[float],
    *,
    cue: str = "appearance",
) -> CueDistribution | None:
    """Fit one camera-pair's same/different distribution with the shared floors.

    Returns ``None`` when either side is too thin — the caller must abstain for
    that pair rather than borrow another pair's colour statistics.
    """

    same_fit = _robust_gaussian(same_values)
    diff_fit = _robust_gaussian(diff_values)
    if same_fit is None or diff_fit is None:
        return None
    floor = _SIGMA_FLOORS.get(cue, _MIN_SIGMA)
    mu_same, sigma_same = same_fit[0], max(same_fit[1], floor)
    mu_diff, sigma_diff = diff_fit[0], max(diff_fit[1], floor)
    if mu_diff <= mu_same:
        pooled = max(sigma_same, sigma_diff, _MIN_SIGMA)
        mu_diff, sigma_same, sigma_diff = mu_same, pooled, pooled
    return CueDistribution(
        mu_same=mu_same, sigma_same=sigma_same,
        mu_diff=mu_diff, sigma_diff=sigma_diff,
        n_same=len(same_values), n_diff=len(diff_values), fitted=True,
    )
