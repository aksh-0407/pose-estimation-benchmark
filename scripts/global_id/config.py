"""Validated configuration for P4a global tracking and P4b stitching."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from pose_estimation.cricket.ground_kalman import ROLE_PARAMS


def _default_role_params() -> dict[str, dict[str, float]]:
    return {
        role: {
            "alpha": params.alpha,
            "sigma_a": params.sigma_a,
            "measurement_noise": params.measurement_noise,
        }
        for role, params in ROLE_PARAMS.items()
    }


def _default_incompatible_roles() -> list[list[str]]:
    return [
        ["bowler", "wicketkeeper"],
        ["striker", "wicketkeeper"],
        ["bowler", "striker"],
        ["bowler", "non_striker"],
        ["umpire", "bowler"],
        ["umpire", "striker"],
        ["umpire", "wicketkeeper"],
    ]


@dataclass(frozen=True)
class P4AConfig:
    confirm_hits: int = 3
    lost_window_frames: int = 30
    bowler_lost_window_frames: int = 60
    chi2_gate_2dof: float = 5.991
    reentry_temporal_gate_frames: int = 120
    reentry_mahalanobis_gate: float = 5.991
    reentry_gap_scale_frames: float = 60.0
    reentry_kinematic_slack: float = 1.5
    role_latch_frames: int = 5
    cap_max_pos_var: float = 25.0
    confidence_high: float = 0.7
    confidence_discard: float = 0.3
    local_identity_mahalanobis_gate: float = 25.0
    role_params: dict[str, dict[str, float]] = field(default_factory=_default_role_params)

    def __post_init__(self) -> None:
        for name in ("confirm_hits", "lost_window_frames", "bowler_lost_window_frames",
                     "reentry_temporal_gate_frames", "role_latch_frames"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("chi2_gate_2dof", "reentry_mahalanobis_gate", "reentry_gap_scale_frames",
                     "reentry_kinematic_slack", "cap_max_pos_var",
                     "local_identity_mahalanobis_gate"):
            _positive(name, getattr(self, name))
        for name in ("confidence_high", "confidence_discard"):
            _range(name, getattr(self, name), 0.0, 1.0)
        if self.confidence_discard > self.confidence_high:
            raise ValueError("confidence_discard must be <= confidence_high")
        if not isinstance(self.role_params, dict) or "unknown" not in self.role_params:
            raise ValueError("role_params must be a mapping containing unknown")
        for role, values in self.role_params.items():
            if not isinstance(role, str) or not isinstance(values, dict):
                raise ValueError("role_params entries must be role -> mapping")
            unknown = set(values) - {"alpha", "sigma_a", "measurement_noise"}
            if unknown or set(values) != {"alpha", "sigma_a", "measurement_noise"}:
                raise ValueError(f"invalid role_params entry for {role}")
            for key, value in values.items():
                _positive(f"role_params.{role}.{key}", value)


@dataclass(frozen=True)
class P4BConfig:
    enabled: bool = True
    cross_camera_min_frames: int = 30
    cross_camera_min_track_ratio: float = 0.5
    temporal_gate_frames: int = 120
    w_temporal: float = 0.1
    w_spatial: float = 1.0
    w_role: float = 100.0
    new_traj_cost_factor: float = 0.5
    velocity_continuity_weight: float = 0.5
    kinematic_slack: float = 1.5
    incompatible_role_pairs: list[list[str]] = field(default_factory=_default_incompatible_roles)

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if type(self.cross_camera_min_frames) is not int or self.cross_camera_min_frames <= 0:
            raise ValueError("cross_camera_min_frames must be a positive integer")
        _range("cross_camera_min_track_ratio", self.cross_camera_min_track_ratio, 0.0, 1.0)
        if type(self.temporal_gate_frames) is not int or self.temporal_gate_frames <= 0:
            raise ValueError("temporal_gate_frames must be a positive integer")
        for name in ("w_spatial", "new_traj_cost_factor", "kinematic_slack"):
            _positive(name, getattr(self, name))
        for name in ("w_temporal", "w_role", "velocity_continuity_weight"):
            _nonnegative(name, getattr(self, name))
        if not isinstance(self.incompatible_role_pairs, list):
            raise ValueError("incompatible_role_pairs must be a list")
        for pair in self.incompatible_role_pairs:
            if not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, str) for v in pair):
                raise ValueError("incompatible_role_pairs entries must be two role strings")


@dataclass(frozen=True)
class P4Config:
    frame_rate_fps: float = 50.0
    kinematic_v_max_mps: float = 9.0
    p4a: P4AConfig = field(default_factory=P4AConfig)
    p4b: P4BConfig = field(default_factory=P4BConfig)

    def __post_init__(self) -> None:
        _positive("frame_rate_fps", self.frame_rate_fps)
        _positive("kinematic_v_max_mps", self.kinematic_v_max_mps)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def _nonnegative(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative number")


def _range(name: str, value: Any, low: float, high: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be numeric")
    if not low <= float(value) <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")


def _build_nested(cls, raw: Any, section: str):
    if raw is None:
        return cls()
    if not isinstance(raw, dict):
        raise ValueError(f"{section} must be a mapping")
    names = {item.name for item in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown {section} config keys: {sorted(unknown)}")
    return cls(**raw)


def load_global_id_config(path: str | Path | None) -> P4Config:
    if path is None:
        return P4Config()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("global-id config must be a mapping")
    unknown = set(raw) - {"frame_rate_fps", "kinematic_v_max_mps", "p4a", "p4b"}
    if unknown:
        raise ValueError(f"unknown global-id config keys: {sorted(unknown)}")
    return P4Config(
        frame_rate_fps=float(raw.get("frame_rate_fps", 50.0)),
        kinematic_v_max_mps=float(raw.get("kinematic_v_max_mps", 9.0)),
        p4a=_build_nested(P4AConfig, raw.get("p4a"), "p4a"),
        p4b=_build_nested(P4BConfig, raw.get("p4b"), "p4b"),
    )
