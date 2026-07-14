"""Configuration for the 01 (stabilization) 2D stabilization stage.

A small, validated dataclass loaded from ``configs/01_stabilization.yaml``. Unknown
keys are rejected so a typo cannot silently disable a knob (the same contract the P2/P3/P4
config loaders follow).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "p1b_stabilization.yaml"


@dataclass(frozen=True)
class LinkConfig:
    iou_min: float = 0.3
    max_gap_frames: int = 2


@dataclass(frozen=True)
class SmoothingConfig:
    method: str = "one_euro"
    min_cutoff: float = 1.7
    beta: float = 0.30
    d_cutoff: float = 1.0


@dataclass(frozen=True)
class GatingConfig:
    confidence_min: float = 0.30
    max_jump_bbox_frac: float = 0.5
    max_jump_px: float = 120.0


@dataclass(frozen=True)
class StabilizationConfig:
    enabled: bool = True
    frame_rate_fps: float = 50.0
    smooth_native: bool = True
    link: LinkConfig = field(default_factory=LinkConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _reject_unknown(section: str, payload: dict, allowed: set[str]) -> None:
    extra = set(payload) - allowed
    if extra:
        raise ValueError(f"unknown key(s) in stabilization config [{section}]: {sorted(extra)}")


def load_stabilization_config(path: str | Path | None = None) -> StabilizationConfig:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        if path is None:
            return StabilizationConfig()
        raise FileNotFoundError(f"stabilization config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("stabilization config must be a mapping")

    top_allowed = {"enabled", "frame_rate_fps", "smooth_native", "link", "smoothing", "gating"}
    _reject_unknown("top", raw, top_allowed)

    link_raw = raw.get("link", {}) or {}
    smooth_raw = raw.get("smoothing", {}) or {}
    gate_raw = raw.get("gating", {}) or {}
    _reject_unknown("link", link_raw, {"iou_min", "max_gap_frames"})
    _reject_unknown("smoothing", smooth_raw, {"method", "min_cutoff", "beta", "d_cutoff"})
    _reject_unknown("gating", gate_raw, {"confidence_min", "max_jump_bbox_frac", "max_jump_px"})

    method = str(smooth_raw.get("method", "one_euro"))
    if method != "one_euro":
        raise ValueError(f"unsupported smoothing.method: {method!r} (only 'one_euro')")

    return StabilizationConfig(
        enabled=bool(raw.get("enabled", True)),
        frame_rate_fps=float(raw.get("frame_rate_fps", 50.0)),
        smooth_native=bool(raw.get("smooth_native", True)),
        link=LinkConfig(
            iou_min=float(link_raw.get("iou_min", 0.3)),
            max_gap_frames=int(link_raw.get("max_gap_frames", 2)),
        ),
        smoothing=SmoothingConfig(
            method=method,
            min_cutoff=float(smooth_raw.get("min_cutoff", 1.7)),
            beta=float(smooth_raw.get("beta", 0.30)),
            d_cutoff=float(smooth_raw.get("d_cutoff", 1.0)),
        ),
        gating=GatingConfig(
            confidence_min=float(gate_raw.get("confidence_min", 0.30)),
            max_jump_bbox_frac=float(gate_raw.get("max_jump_bbox_frac", 0.5)),
            max_jump_px=float(gate_raw.get("max_jump_px", 120.0)),
        ),
    )
