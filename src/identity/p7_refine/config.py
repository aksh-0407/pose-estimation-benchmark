"""Configuration for the 07 (refinement) physics-constrained 3D skeleton stage.

A small validated loader (unknown keys rejected, same contract as the 01/02/03 loaders)
that produces a :class:`RefineParams` for :func:`identity.p7_refine.refine.refine_identity_sequence`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from identity.p7_refine.refine import RefineParams

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "configs" / "07_refine.yaml"

_ALLOWED = {
    "enabled", "relift", "vis_conf", "edge_margin_px",
    "conf_floor", "max_gap_frames", "fps", "smoother", "root_cutoff_hz",
    "limb_cutoff_hz", "filter_order", "ma_root_window", "ma_limb_window",
    "limb_smoother", "oe_min_cutoff", "oe_beta", "oe_d_cutoff",
    "face_window", "face_cutoff_hz", "foot_window", "foot_cutoff_hz",
    "mid_window", "mid_cutoff_hz",
    "dev_tol", "clamp_angles", "min_hinge_deg", "max_hinge_deg",
}


def load_refine_config(path: str | Path | None = None) -> RefineParams:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        if path is None:
            return RefineParams()
        raise FileNotFoundError(f"refine config not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("refine config must be a mapping")
    extra = set(raw) - _ALLOWED
    if extra:
        raise ValueError(f"unknown key(s) in refine config: {sorted(extra)}")

    defaults = RefineParams()
    return RefineParams(
        enabled=bool(raw.get("enabled", defaults.enabled)),
        relift=bool(raw.get("relift", defaults.relift)),
        vis_conf=float(raw.get("vis_conf", defaults.vis_conf)),
        edge_margin_px=float(raw.get("edge_margin_px", defaults.edge_margin_px)),
        conf_floor=float(raw.get("conf_floor", defaults.conf_floor)),
        max_gap_frames=int(raw.get("max_gap_frames", defaults.max_gap_frames)),
        fps=float(raw.get("fps", defaults.fps)),
        smoother=str(raw.get("smoother", defaults.smoother)),
        root_cutoff_hz=float(raw.get("root_cutoff_hz", defaults.root_cutoff_hz)),
        limb_cutoff_hz=float(raw.get("limb_cutoff_hz", defaults.limb_cutoff_hz)),
        filter_order=int(raw.get("filter_order", defaults.filter_order)),
        ma_root_window=int(raw.get("ma_root_window", defaults.ma_root_window)),
        ma_limb_window=int(raw.get("ma_limb_window", defaults.ma_limb_window)),
        limb_smoother=str(raw.get("limb_smoother", defaults.limb_smoother)),
        oe_min_cutoff=float(raw.get("oe_min_cutoff", defaults.oe_min_cutoff)),
        oe_beta=float(raw.get("oe_beta", defaults.oe_beta)),
        oe_d_cutoff=float(raw.get("oe_d_cutoff", defaults.oe_d_cutoff)),
        face_window=int(raw.get("face_window", defaults.face_window)),
        face_cutoff_hz=float(raw.get("face_cutoff_hz", defaults.face_cutoff_hz)),
        foot_window=int(raw.get("foot_window", defaults.foot_window)),
        foot_cutoff_hz=float(raw.get("foot_cutoff_hz", defaults.foot_cutoff_hz)),
        mid_window=int(raw.get("mid_window", defaults.mid_window)),
        mid_cutoff_hz=float(raw.get("mid_cutoff_hz", defaults.mid_cutoff_hz)),
        dev_tol=float(raw.get("dev_tol", defaults.dev_tol)),
        clamp_angles=bool(raw.get("clamp_angles", defaults.clamp_angles)),
        min_hinge_deg=float(raw.get("min_hinge_deg", defaults.min_hinge_deg)),
        max_hinge_deg=float(raw.get("max_hinge_deg", defaults.max_hinge_deg)),
    )
