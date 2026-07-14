"""Validated configuration for P5 role assignment."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RoleAssignerConfig:
    frame_rate_fps: float = 50.0
    min_track_frames: int = 60
    bowler_min_speed_mps: float = 3.5
    pitch_halfwidth_m: float = 2.5
    role_assignment_version: str = "v0"
    epoch_frames: int = 40
    role_epoch_latch_count: int = 3
    role_assignment_max_cost: float = 8.0

    def __post_init__(self) -> None:
        _positive("frame_rate_fps", self.frame_rate_fps)
        if type(self.min_track_frames) is not int or self.min_track_frames <= 0:
            raise ValueError("min_track_frames must be a positive integer")
        _positive("bowler_min_speed_mps", self.bowler_min_speed_mps)
        _positive("pitch_halfwidth_m", self.pitch_halfwidth_m)
        if self.role_assignment_version not in ("v0", "v1"):
            raise ValueError("role_assignment_version must be 'v0' or 'v1'")
        if type(self.epoch_frames) is not int or self.epoch_frames <= 0:
            raise ValueError("epoch_frames must be a positive integer")
        if type(self.role_epoch_latch_count) is not int or self.role_epoch_latch_count <= 0:
            raise ValueError("role_epoch_latch_count must be a positive integer")
        _positive("role_assignment_max_cost", self.role_assignment_max_cost)


def _positive(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive number")


def load_role_assigner_config(path: str | Path | None) -> RoleAssignerConfig:
    if path is None:
        return RoleAssignerConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError("role-assigner config must be a mapping")
    names = {item.name for item in fields(RoleAssignerConfig)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"unknown role-assigner config keys: {sorted(unknown)}")
    return RoleAssignerConfig(**raw)
