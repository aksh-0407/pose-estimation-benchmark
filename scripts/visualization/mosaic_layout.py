"""Calibration-derived mosaic layout — no hardcoded camera ids.

The bowling end flips between overs, so a fixed tile order cannot stay correct.
Everything here is derived per delivery from calibration + the delivery itself:

* **Pitch axis** — from the pitch calibration config: the crease markings run
  across the pitch, so the pitch axis is perpendicular to their principal
  direction (sign-ambiguous until the bowling end is known).
* **Bowling direction** ``p_hat`` — the fastest early-delivery run along the
  pitch axis is the bowler's run-up; its sign orients the axis. Per-camera
  tracklet motion is used (never the fused cross-camera tracks, whose
  inter-camera bias reads as spurious motion). CLI override available.
* **Columns** — one FACING camera pair per column (derived from the calibration
  optical axes, same helper the association stage uses): the end-on pair first,
  then side pairs ordered striker's-end coverage first.
* **Rows** — end-on pair: the camera looking WITH the delivery goes on top (the
  broadcast "behind the bowler's arm" view). Side pairs: the camera whose tile
  must be mirrored (see below) goes on top — this keeps each column a facing
  pair and flips the whole layout when the bowling end flips.
* **Mirrors** — a side camera is flipped when the delivery would travel
  left-to-right in its tile, so every side tile reads right-to-left (the
  production's broadcast convention). End-on cameras never flip (the delivery
  runs along their depth axis).
* **Bottom row** — the unpaired camera (the pano) sits bottom-middle, flanked
  by the delivery-monitor and roster panels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from pose_estimation.cricket.geometry import camera_axis_lookat, derive_facing_pairs

MONITOR_SLOT = "__monitor__"
ROSTER_SLOT = "__roster__"

_END_ON_MIN_AXIS_DOT = 0.7   # |forward . pitch axis| above this = end-on camera
_MIRROR_COS_THRESHOLD = 0.5  # |cos(image-x, p_hat)| above this = decisive reading


@dataclass(frozen=True)
class MosaicLayout:
    grid: tuple[tuple[str | None, ...], ...]  # rows of camera ids / panel slots
    mirrored: frozenset[str]
    bowling_direction_xy: tuple[float, float] | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> str:
        rows = [" | ".join(slot or "-" for slot in row) for row in self.grid]
        mirror = ", ".join(sorted(self.mirrored)) or "none"
        return "; ".join(rows) + f"  (mirrored: {mirror})"


def load_pitch_axis(pitch_config_path: str | Path) -> np.ndarray | None:
    """Unit pitch axis (sign-ambiguous) from the pitch calibration config.

    Preferred: the two stump mid-base points (``fsmb``/``nsmb``) define the axis
    exactly. Fallback: the principal direction of all marking points — for a
    full-pitch cloud (>= 8 m extent) that IS the pitch axis; for a single-crease
    patch the markings run across the pitch, so the perpendicular is used.
    """

    path = Path(pitch_config_path)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

    def as_xy(value) -> np.ndarray | None:
        try:
            arr = np.asarray(value, dtype=float).ravel()
        except (TypeError, ValueError):
            return None  # metadata fields live beside the marking points
        if arr.shape[0] >= 2 and np.isfinite(arr[:2]).all():
            return arr[:2]
        return None

    far = as_xy(payload.get("fsmb"))
    near = as_xy(payload.get("nsmb"))
    if far is not None and near is not None:
        axis = far - near
        norm = float(np.linalg.norm(axis))
        if norm > 1.0:  # stumps are ~20 m apart; anything tiny is degenerate
            return axis / norm

    points = [xy for value in payload.values() if (xy := as_xy(value)) is not None]
    if len(points) < 3:
        return None
    data = np.asarray(points) - np.mean(points, axis=0)
    _eigvals, eigvecs = np.linalg.eigh(data.T @ data)
    major = eigvecs[:, -1]
    spread = data @ major
    if float(spread.max() - spread.min()) >= 8.0:
        axis = major  # full-pitch cloud: major axis runs down the pitch
    else:
        axis = np.array([-major[1], major[0]])  # crease patch: use perpendicular
    norm = float(np.linalg.norm(axis))
    return axis / norm if norm > 1e-9 else None


def infer_bowling_direction(
    tracklet_series: dict[str, list[tuple[int, np.ndarray]]],
    pitch_axis: np.ndarray,
    *,
    window_frames: int = 50,
    early_fraction: float = 0.5,
    min_speed_mps: float = 3.0,
    frame_rate_fps: float = 50.0,
) -> np.ndarray | None:
    """Signed bowling direction from the fastest early run ALONG the pitch axis.

    ``tracklet_series`` must be per-camera tracklets (single-camera positions):
    fused cross-camera tracks carry inter-camera bias that reads as motion.
    Displacements are projected onto the pitch axis, so cross-pitch noise cannot
    vote. Returns +/- pitch_axis, or ``None`` when nothing runs fast enough.
    """

    axis = np.asarray(pitch_axis, dtype=float)
    axis = axis / max(float(np.linalg.norm(axis)), 1e-9)
    all_frames = [frame for series in tracklet_series.values() for frame, _ in series]
    if not all_frames:
        return None
    cutoff = min(all_frames) + early_fraction * (max(all_frames) - min(all_frames))

    best_speed, best_sign = 0.0, 0.0
    for series in tracklet_series.values():
        ordered = sorted(series, key=lambda item: item[0])
        for i, (frame_a, point_a) in enumerate(ordered):
            if frame_a > cutoff:
                break
            for frame_b, point_b in ordered[i + 1:]:
                gap = frame_b - frame_a
                if gap > window_frames:
                    break
                if gap < window_frames // 2:
                    continue
                along = float((np.asarray(point_b) - np.asarray(point_a)) @ axis)
                speed = abs(along) * frame_rate_fps / gap
                if speed > best_speed:
                    best_speed, best_sign = speed, float(np.sign(along))
    if best_sign == 0.0 or best_speed < min_speed_mps:
        return None
    return axis * best_sign


def _image_x_cosine(projection: np.ndarray, ground_point: np.ndarray,
                    direction_xy: np.ndarray) -> float:
    """cos between image-x and the world ``direction_xy`` at ``ground_point``."""

    base = np.asarray([ground_point[0], ground_point[1], 0.0, 1.0], dtype=float)
    step = np.asarray([direction_xy[0], direction_xy[1], 0.0, 0.0], dtype=float)
    p0 = projection @ base
    p1 = projection @ (base + 0.5 * step)
    if abs(p0[2]) < 1e-9 or abs(p1[2]) < 1e-9:
        return 0.0
    delta = p1[:2] / p1[2] - p0[:2] / p0[2]
    norm = float(np.linalg.norm(delta))
    if norm < 1e-9:
        return 0.0
    return float(delta[0] / norm)


def derive_mosaic_layout(
    projections: dict[str, np.ndarray],
    *,
    bowling_direction_xy: np.ndarray | None = None,
    bowling_end_cam: str | None = None,
) -> MosaicLayout:
    """Build the 3x3 mosaic layout from calibration + the delivery direction."""

    notes: list[str] = []
    cameras = sorted(projections)
    geo = {cam: camera_axis_lookat(projections[cam]) for cam in cameras}
    pairs = derive_facing_pairs(projections)
    paired = {cam for pair in pairs for cam in pair}
    unpaired = [cam for cam in cameras if cam not in paired]

    direction = None
    if bowling_end_cam is not None:
        if bowling_end_cam not in geo:
            raise ValueError(f"unknown bowling_end_cam: {bowling_end_cam}")
        forward = geo[bowling_end_cam][1]
        direction = np.asarray([forward[0], forward[1]], dtype=float)
        notes.append(f"bowling direction from override camera {bowling_end_cam}")
    elif bowling_direction_xy is not None:
        direction = np.asarray(bowling_direction_xy, dtype=float)[:2]
        notes.append("bowling direction inferred from tracklet motion")
    if direction is None or float(np.linalg.norm(direction)) < 1e-9:
        if pairs:
            forward = geo[pairs[0][0]][1]
            direction = np.asarray([forward[0], forward[1]], dtype=float)
            notes.append(f"bowling direction UNKNOWN - fell back to {pairs[0][0]} axis")
        else:
            direction = np.asarray([1.0, 0.0])
            notes.append("bowling direction UNKNOWN - no facing pairs")
    p_hat = direction / max(float(np.linalg.norm(direction)), 1e-9)

    def forward_dot(cam: str) -> float:
        return float(np.asarray(geo[cam][1][:2]) @ p_hat)

    def image_x_cos(cam: str) -> float:
        lookat = geo[cam][2]
        if not np.isfinite(lookat).all():
            return 0.0
        return _image_x_cosine(projections[cam], lookat, p_hat)

    def is_end_on(pair: tuple[str, str]) -> bool:
        return all(abs(forward_dot(cam)) >= _END_ON_MIN_AXIS_DOT for cam in pair)

    def lookat_along(pair: tuple[str, str]) -> float:
        lookats = [geo[cam][2] for cam in pair if np.isfinite(geo[cam][2]).all()]
        if not lookats:
            return float("-inf")
        return float(np.mean(np.asarray(lookats) @ p_hat))

    # Columns: end-on pair(s) first, then side pairs covering the striker's end
    # (farther along the delivery) before those covering the bowling end.
    ordered_pairs = sorted(
        pairs,
        key=lambda pair: (0 if is_end_on(pair) else 1, -lookat_along(pair), pair),
    )

    # Mirrors: every side tile must read the delivery RIGHT-TO-LEFT (broadcast
    # convention chosen for this production), so side cameras where the ball
    # would travel left-to-right get flipped. End-on tiles never flip (the
    # delivery runs along their depth axis).
    mirrored: set[str] = set()
    for cam in cameras:
        if abs(forward_dot(cam)) >= _END_ON_MIN_AXIS_DOT:
            continue
        if image_x_cos(cam) > _MIRROR_COS_THRESHOLD:
            mirrored.add(cam)

    top_row: list[str | None] = []
    mid_row: list[str | None] = []
    for pair in ordered_pairs:
        if is_end_on(pair):
            top = max(pair, key=forward_dot)  # looks WITH the delivery
        else:
            flipped = [cam for cam in pair if cam in mirrored]
            top = flipped[0] if len(flipped) == 1 else min(pair)
            if len(flipped) != 1:
                notes.append(f"side pair {pair} mirror state ambiguous - row order arbitrary")
        bottom = pair[0] if top == pair[1] else pair[1]
        top_row.append(top)
        mid_row.append(bottom)
    while len(top_row) < 3:
        top_row.append(None)
        mid_row.append(None)

    bottom_row: list[str | None] = [MONITOR_SLOT, None, ROSTER_SLOT]
    if unpaired:
        bottom_row[1] = unpaired[0]
        for extra in unpaired[1:]:
            notes.append(f"unpaired camera {extra} has no tile - omitted")

    return MosaicLayout(
        grid=(tuple(top_row[:3]), tuple(mid_row[:3]), tuple(bottom_row)),
        mirrored=frozenset(mirrored),
        bowling_direction_xy=(float(p_hat[0]), float(p_hat[1])),
        notes=tuple(notes),
    )
