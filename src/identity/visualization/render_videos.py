#!/usr/bin/env python3
"""Render Phase 1 cricket pose detections as MP4 videos."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional in minimal envs.
    tqdm = None

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from core.dataset import (
    parse_prediction_filename,
    repo_relative,
    resolve_delivery_camera_dirs,
)
from identity.common.geometry import (
    ground_point_visible_in,
    project_ground_to_pixel,
)
from core.inference.phase1_outputs import COCO_17_EDGES
from identity.visualization.identity_colors import (
    IDENTITY_PALETTE,
    color_for_global_id,
    color_for_player,
)
from identity.visualization.mosaic_layout import (
    MONITOR_SLOT,
    ROSTER_SLOT,
    MosaicLayout,
    derive_mosaic_layout,
    infer_bowling_direction,
    load_pitch_axis,
)


BALL_TRAIL_FRAMES = 18
BALL_COLOR = (0, 215, 255)
PLAYER_COLORS = list(IDENTITY_PALETTE)  # backward-compatible export for callers
EDGE_COLORS = [
    (82, 229, 255),
    (82, 229, 255),
    (254, 189, 96),
    (254, 189, 96),
    (119, 255, 165),
    (118, 205, 255),
    (255, 142, 142),
    (190, 169, 255),
    (118, 205, 255),
    (118, 205, 255),
    (255, 142, 142),
    (255, 142, 142),
    (255, 255, 255),
    (255, 255, 255),
    (209, 209, 209),
    (209, 209, 209),
]


@dataclass(frozen=True)
class VideoSettings:
    fps: float
    keypoint_threshold: float
    per_camera_size: tuple[int, int]
    mosaic_size: tuple[int, int]
    sample_every: int
    max_frames: int | None
    crf: int
    preset: str
    use_ffmpeg: bool
    ball_trail_frames: int
    show: str
    letterbox_tiles: bool = False
    body_paint: bool = True


import functools


@functools.lru_cache(maxsize=1)
def _ffmpeg_has_nvenc() -> bool:
    """True when this ffmpeg build exposes the NVIDIA NVENC H.264 encoder."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return False
    return "h264_nvenc" in out


class VideoSink:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
        use_ffmpeg: bool,
    ) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.process: subprocess.Popen | None = None
        self.writer: cv2.VideoWriter | None = None
        self.encoder = "opencv/mp4v"  # actual encoder used (set below); reported in the manifest
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if use_ffmpeg and shutil.which("ffmpeg"):
            common = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
            ]
            if _ffmpeg_has_nvenc():
                # GPU (NVENC) encode: offloads H.264 to the RTX. -cq maps CRF-like
                # quality; p4/hq is a balanced quality preset. Falls back automatically
                # if NVENC is unavailable at runtime (encoder errors -> close() raises).
                codec = [
                    "-vcodec", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                    "-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p",
                ]
                self.encoder = "ffmpeg/h264_nvenc"
            else:
                codec = ["-vcodec", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
                self.encoder = "ffmpeg/libx264"
            self.process = subprocess.Popen([*common, *codec, str(path)], stdin=subprocess.PIPE)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(
                f"frame shape {frame.shape[:2]} does not match {(self.height, self.width)}"
            )
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            assert self.writer is not None
            self.writer.write(frame)

    def close(self) -> None:
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            return_code = self.process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed for {self.path} with code {return_code}")
        if self.writer is not None:
            self.writer.release()

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="drive")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--delivery-id",
        default=None,
        help="Delivery to render. Optional when the run holds a single delivery.",
    )
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Output dir for videos (default: <run-dir>/visualizations/videos).",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--per-camera-size", default="1280x720")
    parser.add_argument("--mosaic-size", default="1920x1080")
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--keypoint-threshold", type=float, default=0.2)
    parser.add_argument("--ball-trail-frames", type=int, default=BALL_TRAIL_FRAMES)
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument(
        "--bowling-end-cam",
        default=None,
        help=(
            "Camera looking WITH the delivery (behind the bowler's arm). "
            "Overrides the automatic bowling-end inference for the mosaic layout."
        ),
    )
    parser.add_argument(
        "--roles-path",
        default=None,
        help="P5 roles.json (default: probes <run-dir>/../p5/roles.json then <run-dir>/p5/roles.json)",
    )
    parser.add_argument("--mode", choices=["all", "per-camera", "mosaic", "ground"], default="all")
    parser.add_argument("--show", choices=["p2", "p3", "p4"], default=None)
    parser.add_argument("--no-body-paint", dest="body_paint", action="store_false",
                        help="Disable the skeleton body-paint identity overlay "
                             "(W7-RENDER; on by default)")
    parser.add_argument("--letterbox-tiles", action="store_true",
                        help="Aspect-correct wide tiles (C07) by bottom-padding instead of stretching (F2/R-1).")
    parser.add_argument("--crf", type=int, default=22)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--no-ffmpeg", dest="use_ffmpeg", action="store_false")
    parser.set_defaults(use_ffmpeg=True)
    return parser.parse_args()


def parse_size(value: str) -> tuple[int, int]:
    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}, expected WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}, dimensions must be positive")
    return width, height


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def stage_from_manifest(manifest: dict[str, Any]) -> str:
    return {
        "per_camera_tracking": "p2",
        "cross_camera_association": "p3",
        "global_id_tracking": "p4",
        "multi_view_3d_lift": "p4",
    }.get(manifest.get("task"), "p4")


def load_cluster_badges(path: Path) -> dict[tuple[int, str], dict[int, int]]:
    badges: dict[tuple[int, str], dict[int, int]] = {}
    if not path.exists():
        return badges
    for row in iter_jsonl(path):
        frame_index = int(row["frame_index"])
        for cluster in row.get("clusters", []):
            for member in cluster.get("members", []):
                key = (frame_index, member["cam_id"])
                badges.setdefault(key, {})[int(member["player_index"])] = int(cluster["cluster_id"])
    return badges


def load_pitch_extents(path: Path) -> tuple[float, float, float, float]:
    if not path.exists():
        return (-15.0, 15.0, -30.0, 30.0)
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = np.asarray(
        [value[:2] for value in payload.values() if isinstance(value, list) and len(value) >= 2],
        dtype=float,
    )
    if points.size == 0 or not np.isfinite(points).all():
        return (-15.0, 15.0, -30.0, 30.0)
    x_min, y_min = np.min(points, axis=0)
    x_max, y_max = np.max(points, axis=0)
    x_pad = max(3.0, 0.08 * (x_max - x_min))
    y_pad = max(3.0, 0.08 * (y_max - y_min))
    return (float(x_min - x_pad), float(x_max + x_pad), float(y_min - y_pad), float(y_max + y_pad))


def load_ball_positions(events_root: Path, delivery_id: str) -> dict[str, tuple[float, float]]:
    """Map frame_name -> (cx, cy) normalized ball center from the 2D event artifacts."""
    positions: dict[str, tuple[float, float]] = {}
    if not events_root.exists():
        return positions
    best_conf: dict[str, float] = {}
    for delivery_dir in sorted(events_root.glob(f"{delivery_id}_*")):
        two_d = delivery_dir / f"{delivery_dir.name}_2D.json"
        if not two_d.exists():
            continue
        payload = json.loads(two_d.read_text(encoding="utf-8"))
        for frame in payload.get("frames", []):
            for camera in frame.get("cameras", []):
                frame_name = camera.get("frame_name")
                if not frame_name:
                    continue
                for detection in camera.get("detections", []):
                    coords = detection.get("coords")
                    if not coords or len(coords) < 2:
                        continue
                    conf = float(detection.get("confidence_score") or 0.0)
                    if frame_name not in best_conf or conf > best_conf[frame_name]:
                        best_conf[frame_name] = conf
                        positions[frame_name] = (float(coords[0]), float(coords[1]))
    return positions


def ball_trail_for(
    records: list[dict[str, Any]],
    index: int,
    positions: dict[str, tuple[float, float]],
    trail_frames: int,
) -> list[tuple[float, float]]:
    if trail_frames <= 0 or not positions:
        return []
    trail: list[tuple[float, float]] = []
    start = max(0, index - trail_frames + 1)
    for cursor in range(start, index + 1):
        position = positions.get(records[cursor].get("frame_name"))
        if position is not None:
            trail.append(position)
    return trail


def draw_ball_trail(
    image: np.ndarray,
    trail: list[tuple[float, float]],
    *,
    compact: bool,
) -> None:
    if not trail:
        return
    height, width = image.shape[:2]
    points = [(int(round(cx * width)), int(round(cy * height))) for cx, cy in trail]
    line_thickness = 1 if compact else 2
    count = len(points)
    for cursor in range(1, count):
        fade = cursor / count
        color = (0, int(170 + 85 * fade), 255)
        cv2.line(image, points[cursor - 1], points[cursor], color, line_thickness, cv2.LINE_AA)
    cx, cy = points[-1]
    radius = 4 if compact else 6
    cv2.circle(image, (cx, cy), radius + 2, (10, 14, 20), -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)


def load_records(path: Path, *, sample_every: int, max_frames: int | None) -> list[dict[str, Any]]:
    records = []
    for index, record in enumerate(iter_jsonl(path)):
        if index % sample_every != 0:
            continue
        records.append(record)
        if max_frames is not None and len(records) >= max_frames:
            break
    return records


def blend_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1 = max(0, min(image.shape[1], x1))
    x2 = max(0, min(image.shape[1], x2))
    y1 = max(0, min(image.shape[0], y1))
    y2 = max(0, min(image.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = image[y1:y2, x1:x2].copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, image[y1:y2, x1:x2], 1.0 - alpha, 0, image[y1:y2, x1:x2])


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int] = (245, 248, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_chip(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float = 0.48,
) -> tuple[int, int]:
    x, y = origin
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad_x, pad_y = 10, 7
    blend_rect(
        image,
        (x, y - text_h - pad_y),
        (x + text_w + 2 * pad_x, y + pad_y),
        (18, 24, 32),
        0.82,
    )
    cv2.rectangle(
        image,
        (x, y - text_h - pad_y),
        (x + text_w + 2 * pad_x, y + pad_y),
        color,
        1,
        cv2.LINE_AA,
    )
    draw_text(image, text, (x + pad_x, y), scale=scale, color=(248, 250, 255), thickness=1)
    return x + text_w + 2 * pad_x + 8, y


def scale_player(player: dict[str, Any], *, sx: float, sy: float) -> dict[str, Any]:
    scaled = dict(player)
    x, y, w, h = player["bbox_xywh_px"]
    scaled["bbox_xywh_px"] = [x * sx, y * sy, w * sx, h * sy]
    pose = dict(player["pose_2d"])
    pose["keypoints_px"] = [[point[0] * sx, point[1] * sy] for point in pose["keypoints_px"]]
    scaled["pose_2d"] = pose
    return scaled


def mirror_record(record: dict[str, Any], source_width: int) -> dict[str, Any]:
    """Mirror all player geometry about the vertical axis of the SOURCE frame.

    Overlays (chips, labels, skeletons) are drawn AFTER the image is flipped,
    on mirrored coordinates — so text always renders upright and unmirrored.
    """

    mirrored = dict(record)
    players = []
    for player in record.get("players", []):
        copy = dict(player)
        bbox = player.get("bbox_xywh_px")
        if bbox is not None:
            x, y, w, h = bbox
            copy["bbox_xywh_px"] = [source_width - x - w, y, w, h]
        pose = player.get("pose_2d")
        if pose is not None:
            pose_copy = dict(pose)
            pose_copy["keypoints_px"] = [
                [source_width - point[0], point[1]] for point in pose["keypoints_px"]
            ]
            copy["pose_2d"] = pose_copy
        players.append(copy)
    mirrored["players"] = players
    return mirrored


def confidence_color(score: float) -> tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    if score >= 0.75:
        return (111, 255, 151)
    if score >= 0.45:
        return (82, 220, 255)
    return (78, 118, 255)


COCO_TORSO = (5, 6, 12, 11)  # L-shoulder, R-shoulder, R-hip, L-hip (quad order)
COCO_HEAD = (0, 1, 2, 3, 4)  # nose, eyes, ears


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def place_chip(
    bbox: tuple[int, int, int, int],
    chip_wh: tuple[int, int],
    image_wh: tuple[int, int],
    reserves: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
) -> tuple[int, int, bool]:
    """Collision-aware chip placement (W7-RENDER).

    Candidates in preference order: above, below, right-of-box, left-of-box, then
    stacked upward above the box (leader line advised). Returns (x, y_baseline,
    stacked) for the first candidate that avoids every previously placed chip and
    the tile's header/footer reserves; falls back to the legacy inside-the-box spot
    when everything collides (still recorded, so later chips dodge it).
    """

    x, y, x2, y2 = bbox
    chip_w, chip_h = chip_wh
    width_limit, height_limit = image_wh
    top_clear, bottom_clear = reserves

    def rect(cx: int, cy_baseline: int) -> tuple[int, int, int, int]:
        return (cx, cy_baseline - chip_h, cx + chip_w, cy_baseline)

    base_x = int(np.clip(x, 6, max(6, width_limit - chip_w - 6)))
    candidates: list[tuple[int, int, bool]] = [
        (base_x, y - 8, False),                          # above
        (base_x, y2 + 10 + chip_h, False),               # below
        (min(x2 + 6, width_limit - chip_w - 6), y + chip_h + 4, False),   # right of box
        (max(6, x - chip_w - 6), y + chip_h + 4, False),                  # left of box
    ]
    for level in range(1, 5):                            # stacked above, rising
        candidates.append((base_x, y - 8 - level * (chip_h + 4), True))
    for level in range(1, 5):                            # stacked below, descending
        candidates.append((base_x, y2 + 10 + chip_h + level * (chip_h + 4), True))
    for level in range(1, 5):                            # stacked beside, right column
        candidates.append(
            (min(x2 + 6, width_limit - chip_w - 6), y + level * (chip_h + 6), True)
        )
    for cx, cy, stacked in candidates:
        if cy - chip_h < top_clear or cy > bottom_clear:
            continue
        r = rect(cx, cy)
        if any(_rects_overlap(r, other) for other in placed):
            continue
        placed.append(r)
        return cx, cy, stacked
    cy = max(top_clear + chip_h, min(y + chip_h + 4, bottom_clear))  # legacy inside
    placed.append(rect(base_x, cy))
    return base_x, cy, False


def draw_body_paint(
    overlay: np.ndarray,
    players: list[dict[str, Any]],
    colors: list[tuple[int, int, int]],
    keypoint_threshold: float,
) -> None:
    """Skeleton body-paint (W7-RENDER, user-approved style): torso quad + limb
    capsules + head disc filled with the identity colour on a shared overlay layer
    (blended once by the caller). Falls back to nothing when torso joints are
    missing — the bbox outline still identifies the player."""

    for player, color in zip(players, colors):
        keypoints = player["pose_2d"]["keypoints_px"]
        confidence = player["pose_2d"]["confidence"]
        if len(confidence) < 17:
            continue
        bbox_h = max(1.0, float(player["bbox_xywh_px"][3]))
        limb_w = max(4, int(bbox_h) // 7)

        def pt(i):
            return (int(round(keypoints[i][0])), int(round(keypoints[i][1])))

        torso = [pt(i) for i in COCO_TORSO if float(confidence[i]) >= keypoint_threshold]
        if len(torso) == 4:
            cv2.fillConvexPoly(overlay, np.asarray(torso, dtype=np.int32), color, cv2.LINE_AA)
        for left, right in COCO_17_EDGES:
            if left >= len(confidence) or right >= len(confidence):
                continue
            if min(float(confidence[left]), float(confidence[right])) < keypoint_threshold:
                continue
            cv2.line(overlay, pt(left), pt(right), color, limb_w, cv2.LINE_AA)
        head = [pt(i) for i in COCO_HEAD if float(confidence[i]) >= keypoint_threshold]
        if head:
            cx = int(round(sum(h[0] for h in head) / len(head)))
            cy = int(round(sum(h[1] for h in head) / len(head)))
            cv2.circle(overlay, (cx, cy), max(3, int(bbox_h) // 10), color, -1, cv2.LINE_AA)


_ROLE_TAGS = {
    "bowler": "BOWLER",
    "striker": "STRIKER",
    "non_striker": "NON-STRIKER",
    "wicketkeeper": "KEEPER",
    "umpire": "UMPIRE",
}


def draw_players(
    image: np.ndarray,
    record: dict[str, Any],
    *,
    sx: float,
    sy: float,
    keypoint_threshold: float,
    compact: bool,
    show: str,
    roles: dict[str, str] | None = None,
    suppressed: set[str] | None = None,
    body_paint: bool = True,
) -> int:
    players = [
        scale_player(player, sx=sx, sy=sy)
        for player in record.get("players", [])
        if player.get("bbox_xywh_px") is not None and player.get("pose_2d") is not None
        and str(player.get("global_player_id")) not in (suppressed or set())
    ]
    line_thickness = 2 if compact else 3
    point_radius = 3 if compact else 5
    font_scale = 0.42 if compact else 0.56

    def _player_color(player):
        if show == "p3" and player.get("_cluster_id") is not None:
            return color_for_global_id(
                f"cluster:{record['frame_index']}:{player['_cluster_id']}"
            )
        return color_for_player(player.get("global_player_id"), player.get("local_track_id"))

    colors = [_player_color(player) for player in players]
    if body_paint and players:
        overlay = image.copy()
        draw_body_paint(overlay, players, colors, keypoint_threshold)
        cv2.addWeighted(overlay, 0.35, image, 0.65, 0, dst=image)
    # Left-to-right processing makes the collision-aware chip placement stable.
    order = sorted(range(len(players)), key=lambda i: players[i]["bbox_xywh_px"][0])
    placed_chips: list[tuple[int, int, int, int]] = []

    for player_index, order_index in enumerate(order, start=1):
        player = players[order_index]
        color = colors[order_index]
        x, y, w, h = [int(round(value)) for value in player["bbox_xywh_px"]]
        x2, y2 = x + w, y + h
        detection_confidence = player.get("detection_confidence")
        if detection_confidence is None:
            detection_confidence = player.get("track_confidence")
        detection_conf = float(detection_confidence or 0.0)

        if not body_paint:
            blend_rect(image, (x, y), (x2, y2), color, 0.08)
        cv2.rectangle(image, (x, y), (x2, y2), color, line_thickness, cv2.LINE_AA)
        corner = max(12, min(w, h) // 7)
        cv2.line(image, (x, y), (x + corner, y), (255, 255, 255), line_thickness, cv2.LINE_AA)
        cv2.line(image, (x, y), (x, y + corner), (255, 255, 255), line_thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y), (x2 - corner, y), (255, 255, 255), line_thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y), (x2, y + corner), (255, 255, 255), line_thickness, cv2.LINE_AA)

        if show == "p4":
            identity = player.get("global_player_id") or "unassigned"
        elif show == "p3":
            identity = f"C{player.get('_cluster_id')}" if player.get("_cluster_id") is not None else "unmatched"
        else:
            identity = player.get("local_track_id") or f"det-{player_index}"
        # Roles are shown ONLY in the roster panel (user directive) — chips carry
        # identity + confidence, kept short so side-by-side players stay readable.
        label = f"{identity}  trk {float(player.get('track_confidence') or 0.0):.2f}"
        # Collision-aware placement (W7-RENDER): candidates above/below/beside the
        # box, then stacked upward with a leader line — two adjacent players can no
        # longer draw unreadable overlapping chips.
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        chip_w, chip_h = text_w + 20, text_h + 14
        height_limit, width_limit = image.shape[:2]
        top_clear = 52 if compact else 78
        bottom_clear = height_limit - (32 if compact else 42)
        label_x, label_y, stacked = place_chip(
            (x, y, x2, y2), (chip_w, chip_h), (width_limit, height_limit),
            (top_clear, bottom_clear), placed_chips,
        )
        if stacked:
            cv2.line(image, (label_x + 6, label_y + 2), (x + 6, y), color, 1, cv2.LINE_AA)
        next_x, _ = draw_chip(image, label, (label_x, label_y), color=color, scale=font_scale)
        bar_w = 56 if compact else 84
        bar_h = 5 if compact else 7
        bar_x = next_x
        bar_y = label_y - bar_h - 3
        blend_rect(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (48, 54, 64), 0.85)
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + int(round(bar_w * detection_conf)), bar_y + bar_h),
            confidence_color(detection_conf),
            -1,
        )

        keypoints = player["pose_2d"]["keypoints_px"]
        confidence = player["pose_2d"]["confidence"]
        for edge_index, (left, right) in enumerate(COCO_17_EDGES):
            if left >= len(confidence) or right >= len(confidence):
                continue
            edge_conf = min(float(confidence[left]), float(confidence[right]))
            if edge_conf < keypoint_threshold:
                continue
            p1 = tuple(int(round(value)) for value in keypoints[left])
            p2 = tuple(int(round(value)) for value in keypoints[right])
            cv2.line(
                image,
                p1,
                p2,
                EDGE_COLORS[edge_index % len(EDGE_COLORS)],
                max(1, line_thickness),
                cv2.LINE_AA,
            )

        for point, score in zip(keypoints, confidence):
            score = float(score)
            if score < keypoint_threshold:
                continue
            center = tuple(int(round(value)) for value in point)
            cv2.circle(image, center, point_radius + 1, (9, 13, 20), -1, cv2.LINE_AA)
            cv2.circle(image, center, point_radius, confidence_color(score), -1, cv2.LINE_AA)
            cv2.circle(image, center, point_radius, (255, 255, 255), 1, cv2.LINE_AA)

    return len(players)


def add_header(
    image: np.ndarray,
    *,
    title: str,
    subtitle: str,
    right_text: str,
    compact: bool,
) -> None:
    height, width = image.shape[:2]
    header_h = 46 if compact else 68
    blend_rect(image, (0, 0), (width, header_h), (11, 16, 24), 0.78)
    cv2.line(image, (0, header_h), (width, header_h), (82, 220, 255), 1, cv2.LINE_AA)
    draw_text(
        image,
        title,
        (16, 29 if compact else 42),
        scale=0.58 if compact else 0.88,
        color=(250, 252, 255),
        thickness=2 if not compact else 1,
    )
    if not compact:
        draw_text(image, subtitle, (18, 61), scale=0.45, color=(184, 196, 214), thickness=1)
    (right_w, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42 if compact else 0.55, 1)
    draw_text(
        image,
        right_text,
        (max(16, width - right_w - 18), 29 if compact else 43),
        scale=0.42 if compact else 0.55,
        color=(218, 229, 245),
        thickness=1,
    )


def add_footer(image: np.ndarray, *, compact: bool) -> None:
    height, width = image.shape[:2]
    footer_h = 28 if compact else 38
    blend_rect(image, (0, height - footer_h), (width, height), (11, 16, 24), 0.64)
    text = (
        "COCO-17  |  bbox + joint confidence"
        if compact
        else "COCO-17 joints  |  bbox confidence  |  hidden joints suppressed by threshold"
    )
    draw_text(
        image,
        text,
        (16, height - 10),
        scale=0.36 if compact else 0.48,
        color=(198, 208, 222),
        thickness=1,
    )
    legend_x = width - (168 if compact else 230)
    y = height - 13
    for label, color in [("high", (111, 255, 151)), ("mid", (82, 220, 255)), ("low", (78, 118, 255))]:
        cv2.circle(image, (legend_x, y), 4 if compact else 5, color, -1, cv2.LINE_AA)
        draw_text(
            image,
            label,
            (legend_x + 9, y + 4),
            scale=0.34 if compact else 0.42,
            color=(213, 222, 235),
            thickness=1,
        )
        legend_x += 49 if compact else 66


def render_feed_frame(
    image: np.ndarray,
    record: dict[str, Any],
    *,
    output_size: tuple[int, int],
    keypoint_threshold: float,
    compact: bool,
    ball_trail: list[tuple[float, float]] | None = None,
    show: str = "p4",
    mirror: bool = False,
    ghosts: list[Ghost] | None = None,
    letterbox: bool = False,
    roles: dict[str, str] | None = None,
    suppressed: set[str] | None = None,
    body_paint: bool = True,
) -> np.ndarray:
    output_width, output_height = output_size
    source_height, source_width = image.shape[:2]
    if letterbox:
        # F2/R-1: a heterogeneous camera (C07 is ~3775x960) stretched into a 16:9
        # tile distorts the picture. Pad the BOTTOM of the source to the tile
        # aspect before the resize — coordinates are untouched (padding is below
        # every player), the resize becomes aspect-preserving, and overlays stay
        # aligned with the (now undistorted) picture.
        target_height = int(round(source_width * output_height / output_width))
        if target_height > source_height:
            pad = np.zeros((target_height - source_height, source_width, 3), dtype=image.dtype)
            image = np.concatenate([image, pad], axis=0)
            source_height = target_height
    frame = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    if mirror:
        # Flip the PICTURE first, then draw every overlay on mirrored
        # coordinates — text and chips always render upright.
        frame = cv2.flip(frame, 1)
        record = mirror_record(record, source_width)
        if ball_trail:
            ball_trail = [(1.0 - cx, cy) for cx, cy in ball_trail]
        if ghosts:
            ghosts = [
                ghost._replace(foot=(source_width - ghost.foot[0], ghost.foot[1]))
                for ghost in ghosts
            ]
    sx = output_width / source_width
    sy = output_height / source_height
    player_count = draw_players(
        frame,
        record,
        sx=sx,
        sy=sy,
        keypoint_threshold=keypoint_threshold,
        compact=compact,
        show=show,
        roles=roles,
        suppressed=suppressed,
        body_paint=body_paint,
    )
    if ghosts:
        avoid_rects = [
            (
                int(player["bbox_xywh_px"][0] * sx),
                int(player["bbox_xywh_px"][1] * sy),
                int((player["bbox_xywh_px"][0] + player["bbox_xywh_px"][2]) * sx),
                int((player["bbox_xywh_px"][1] + player["bbox_xywh_px"][3]) * sy),
            )
            for player in record.get("players", [])
            if player.get("bbox_xywh_px") is not None
        ]
        draw_ghost_markers(frame, ghosts, sx=sx, sy=sy, avoid_rects=avoid_rects)
    if ball_trail:
        draw_ball_trail(frame, ball_trail, compact=compact)
    camera_title = record["camera_id"].replace("_", " ").upper()
    if mirror:
        camera_title += "  (mirrored)"
    if compact:
        title = camera_title
        subtitle = ""
        right = f"players {player_count}"
    else:
        title = f"{camera_title}  |  {record['delivery_id']}"
        subtitle = f"frame {record['frame_index']}  |  source {source_width}x{source_height}"
        right = f"players {player_count}"
    add_header(frame, title=title, subtitle=subtitle, right_text=right, compact=compact)
    add_footer(frame, compact=compact)
    return frame


class Ghost(NamedTuple):
    """A reprojected marker for a player NOT detected in this camera this frame.

    ``status`` is ``"occluded"`` (still tracked by another camera — the fused
    position is current) or ``"lost"`` (undetected in every camera — drawn from the
    last-known fused position, so its ground area can be inspected wherever it is
    visible). ``alpha`` in [0, 1] fades the marker as the last observation ages.
    """

    player_id: str
    foot: tuple[float, float]
    status: str = "occluded"
    alpha: float = 1.0


def draw_ghost_markers(
    image: np.ndarray,
    ghosts: list[Ghost],
    *,
    sx: float,
    sy: float,
    avoid_rects: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """Mark players undetected in THIS camera but placed by the tracker elsewhere.

    An ``occluded`` ghost is a player another camera still sees (the detector
    produced nothing here because someone stands in front); a ``lost`` ghost is a
    player no camera sees this frame, drawn from its last-known fused ground position
    so a viewer can see exactly where a disappeared id "should" be in every camera
    that frames that ground. Viewer aid only — no synthetic detection enters the
    pipeline. The chip dodges detection boxes and earlier ghost chips so it never
    covers a visible player.
    """

    height, width = image.shape[:2]
    occupied: list[tuple[int, int, int, int]] = list(avoid_rects or [])

    def overlaps(rect: tuple[int, int, int, int]) -> bool:
        rx1, ry1, rx2, ry2 = rect
        for ox1, oy1, ox2, oy2 in occupied:
            if rx1 < ox2 and rx2 > ox1 and ry1 < oy2 and ry2 > oy1:
                return True
        return False

    def _fade(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
        # Fade toward the dark tile background so an aging ghost visibly dims.
        bg = np.array((10, 14, 21), dtype=float)
        blended = bg + (np.array(color, dtype=float) - bg) * float(np.clip(alpha, 0.15, 1.0))
        return tuple(int(round(v)) for v in blended)

    for ghost in sorted(ghosts):
        player_id, (foot_x, foot_y) = ghost.player_id, ghost.foot
        x, y = int(round(foot_x * sx)), int(round(foot_y * sy))
        if not (0 <= x < width and 0 <= y < height):
            continue
        base = color_for_global_id(player_id)
        lost = ghost.status == "lost"
        # A "lost" ghost is desaturated (greyed) toward its id colour so it reads as
        # "no camera sees this" without losing the id association; occluded stays vivid.
        color = _fade(base if not lost else tuple((c + 150) // 2 for c in base), ghost.alpha)
        ring = (18, 7) if not lost else (20, 8)
        cv2.ellipse(image, (x, y), ring, 0, 0, 360, (12, 16, 24), 3, cv2.LINE_AA)
        cv2.ellipse(image, (x, y), ring, 0, 0, 360, color, 1, cv2.LINE_AA)
        step = 45 if not lost else 30
        for angle in range(0, 360, step):  # dashed inner ring: reads as "not a detection"
            cv2.ellipse(image, (x, y), (11, 4), 0, angle, angle + step // 2, color, 1, cv2.LINE_AA)
        if lost:
            cv2.drawMarker(image, (x, y), color, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)

        label = f"{player_id}  {ghost.status}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        chip_w, chip_h = text_w + 20, text_h + 14
        candidates = (
            (x - chip_w // 2, y + 14 + text_h),          # below the ellipse
            (x - chip_w // 2, y - 18),                   # above
            (x + 24, y + 4),                             # right
            (x - chip_w - 24, y + 4),                    # left
        )
        chosen = None
        for cx, cy in candidates:
            cx = int(np.clip(cx, 6, max(6, width - chip_w - 6)))
            cy = int(np.clip(cy, 52 + chip_h, height - 34))
            rect = (cx, cy - chip_h, cx + chip_w, cy + 6)
            if not overlaps(rect):
                chosen = (cx, cy, rect)
                break
        if chosen is None:  # everything is crowded: fall back to below, clamped
            cx = int(np.clip(x - chip_w // 2, 6, max(6, width - chip_w - 6)))
            cy = int(np.clip(y + 14 + text_h, 52 + chip_h, height - 34))
            chosen = (cx, cy, (cx, cy - chip_h, cx + chip_w, cy + 6))
        cx, cy, rect = chosen
        occupied.append(rect)
        draw_chip(image, label, (cx, cy), color=color, scale=0.4)


def cam_short(camera_id: str) -> str:
    try:
        return f"C{int(camera_id.rsplit('_', 1)[1])}"
    except (ValueError, IndexError):
        return camera_id


def load_suppression(path: Path) -> set[str]:
    """Suppressed global ids from a P5b ``suppression.json``; empty set when absent."""
    if not path.is_file():
        return set()
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return set()
    if not payload.get("enabled"):
        return set()
    return {str(pid) for pid in (payload.get("suppressed") or {})}


def load_roles(path: Path) -> dict[str, str]:
    """Roles from a P5 ``roles.json`` artifact; {} when the phase has not run."""

    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    roles = {}
    for player_id, entry in (payload.get("roles") or {}).items():
        role = (entry or {}).get("role")
        if role and role != "unknown":
            roles[str(player_id)] = str(role)
    return roles


def build_delivery_roster(
    records_by_camera: dict[str, list[dict[str, Any]]],
    roles: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Every global id seen anywhere in the delivery, in first-appearance order."""

    stats: dict[str, dict[str, Any]] = {}
    for records in records_by_camera.values():
        for record in records:
            frame_index = int(record["frame_index"])
            for player in record.get("players", []):
                player_id = player.get("global_player_id")
                if not player_id:
                    continue
                entry = stats.setdefault(
                    str(player_id), {"first": frame_index, "last": frame_index, "frames": 0}
                )
                entry["first"] = min(entry["first"], frame_index)
                entry["last"] = max(entry["last"], frame_index)
                entry["frames"] += 1
    roles = roles or {}
    return [
        {"id": player_id, "role": roles.get(player_id), **entry}
        for player_id, entry in sorted(stats.items(), key=lambda kv: (kv[1]["first"], kv[0]))
    ]


def draw_roster_panel(
    *,
    size: tuple[int, int],
    roster: list[dict[str, Any]],
    visible_now: dict[str, list[str]],
    accent: tuple[int, int, int] = (129, 236, 145),
) -> np.ndarray:
    """Full-delivery roster: every id ever assigned, who is on screen right now
    (and in which cameras), and a reserved ROLE column for the upcoming role
    classifier."""

    width, height = size
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (17, 22, 31)
    blend_rect(panel, (0, 0), (width, height), (28, 36, 48), 0.55)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (58, 68, 82), 1, cv2.LINE_AA)
    cv2.line(panel, (0, 0), (width, 0), accent, 4, cv2.LINE_AA)
    draw_text(panel, "GLOBAL ROSTER", (20, 36), scale=0.62, color=(250, 252, 255), thickness=2)
    live_count = sum(1 for entry in roster if visible_now.get(entry["id"]))
    draw_text(
        panel,
        f"{live_count}/{len(roster)} on screen",
        (width - 150, 34),
        scale=0.42,
        color=(184, 196, 214),
        thickness=1,
    )

    top = 56
    row_height = 24
    rows_per_column = max(1, (height - top - 10) // row_height)
    columns = 1 if len(roster) <= rows_per_column else 2
    column_width = width // columns
    header_color = (140, 152, 170)
    for column in range(columns):
        x0 = 20 + column * column_width
        draw_text(panel, "ID", (x0 + 20, top), scale=0.38, color=header_color, thickness=1)
        draw_text(panel, "LIVE IN", (x0 + 96, top), scale=0.38, color=header_color, thickness=1)
        draw_text(panel, "ROLE", (x0 + column_width - 90, top), scale=0.38, color=header_color, thickness=1)

    for index, entry in enumerate(roster[: rows_per_column * columns]):
        column, row = divmod(index, rows_per_column)
        x0 = 20 + column * column_width
        y = top + 20 + row * row_height
        player_id = entry["id"]
        cams = visible_now.get(player_id) or []
        live = bool(cams)
        color = color_for_global_id(player_id)
        if live:
            blend_rect(panel, (x0 - 8, y - 15), (x0 + column_width - 24, y + 6), (40, 52, 66), 0.55)
        cv2.rectangle(panel, (x0, y - 11), (x0 + 12, y + 1), color, -1, cv2.LINE_AA)
        cv2.rectangle(panel, (x0, y - 11), (x0 + 12, y + 1), (235, 240, 248), 1, cv2.LINE_AA)
        text_color = (245, 248, 255) if live else (120, 132, 150)
        draw_text(panel, player_id, (x0 + 20, y), scale=0.46, color=text_color, thickness=1)
        live_text = " ".join(cam_short(cam) for cam in sorted(cams)) if live else "-"
        draw_text(
            panel,
            live_text,
            (x0 + 96, y),
            scale=0.42,
            color=(148, 236, 164) if live else (96, 106, 122),
            thickness=1,
        )
        # Reserved for the role classifier (kept as a placeholder on purpose).
        draw_text(
            panel,
            entry.get("role") or "-",
            (x0 + column_width - 90, y),
            scale=0.42,
            color=(210, 224, 240),
            thickness=1,
        )
    overflow = len(roster) - rows_per_column * columns
    if overflow > 0:
        draw_text(
            panel,
            f"+{overflow} more",
            (20, height - 12),
            scale=0.4,
            color=(140, 152, 170),
            thickness=1,
        )
    return panel


def draw_info_panel(
    *,
    size: tuple[int, int],
    title: str,
    lines: list[str],
    accent: tuple[int, int, int],
) -> np.ndarray:
    width, height = size
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (17, 22, 31)
    blend_rect(panel, (0, 0), (width, height), (28, 36, 48), 0.55)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (58, 68, 82), 1, cv2.LINE_AA)
    cv2.line(panel, (0, 0), (width, 0), accent, 4, cv2.LINE_AA)
    draw_text(panel, title, (24, 48), scale=0.72, color=(250, 252, 255), thickness=2)
    y = 88
    for line in lines:
        draw_text(panel, line, (24, y), scale=0.46, color=(204, 214, 230), thickness=1)
        y += 34
    return panel


def draw_bev_panel(
    *,
    size: tuple[int, int],
    frame_grounds: dict[str, tuple[float, float]],
    extents: tuple[float, float, float, float],
    frame_index: int,
    trails: dict[str, list[tuple[int, int]]],
    single_ids: set[str] | None = None,
    ghost_grounds: dict[str, tuple[float, float]] | None = None,
    trail_len: int = 120,
) -> np.ndarray:
    """QT-style bird's-eye tile: each player's world (x,y) as an id-coloured dot + trail.

    Replaces the text delivery-monitor tile in the mosaic. ``trails`` is mutated across
    frames so short motion tails persist. ``extents`` is the fixed world window.
    ``ghost_grounds`` are ids currently detected by no camera — drawn as greyed hollow
    markers at their last-known position so a disappearance is visible on the field plot.
    """

    width, height = size
    margin = 30
    header_h = 34
    x_min, x_max, y_min, y_max = extents
    panel = np.full((height, width, 3), (26, 58, 34), dtype=np.uint8)  # deep grass

    # Uniform metric scale (px per world-metre) so the field and its circles are not
    # distorted by the tile aspect ratio — the previous panel stretched them.
    world_w = max(x_max - x_min, 1e-6)
    world_h = max(y_max - y_min, 1e-6)
    avail_w = width - 2 * margin
    avail_h = height - 2 * margin - header_h
    scale = min(avail_w / world_w, avail_h / world_h)
    cxw, cyw = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    origin_px = (width / 2.0, (header_h + height) / 2.0)

    def w2p(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(origin_px[0] + (x - cxw) * scale)),
            int(round(origin_px[1] - (y - cyw) * scale)),
        )

    def w_circle(cx: float, cy: float, radius_m: float, color, thickness: int) -> None:
        cv2.circle(panel, w2p(cx, cy), max(1, int(round(radius_m * scale))), color, thickness, cv2.LINE_AA)

    # Subtle mowing stripes for depth.
    for i in range(0, width, max(24, int(6 * scale))):
        shade = 4 if (i // max(24, int(6 * scale))) % 2 == 0 else -4
        panel[header_h:, i:i + max(12, int(3 * scale))] = np.clip(
            panel[header_h:, i:i + max(12, int(3 * scale))].astype(int) + shade, 0, 255
        ).astype(np.uint8)

    # Field reference geometry (metric): 30-yard ring (27.4 m) + a decorative
    # boundary arc + the pitch strip and popping creases from world coordinates.
    w_circle(0.0, 0.0, 68.0, (58, 96, 66), 2)          # nominal boundary
    w_circle(0.0, 0.0, 27.43, (150, 205, 165), 1)      # 30-yard inner ring
    # Pitch strip (3.05 m x 20.12 m along the y axis) as a filled tan rectangle.
    p1, p2 = w2p(-1.525, 10.06), w2p(1.525, -10.06)
    cv2.rectangle(panel, p1, p2, (150, 180, 200), -1, cv2.LINE_AA)
    blend_rect(panel, (min(p1[0], p2[0]), min(p1[1], p2[1])),
               (max(p1[0], p2[0]), max(p1[1], p2[1])), (120, 150, 175), 0.35)
    cv2.rectangle(panel, p1, p2, (210, 225, 235), 1, cv2.LINE_AA)
    for crease_y in (8.84, -8.84):  # popping creases
        cv2.line(panel, w2p(-1.83, crease_y), w2p(1.83, crease_y), (235, 245, 250), 1, cv2.LINE_AA)
    for stump_y in (10.06, -10.06):  # stumps
        cv2.circle(panel, w2p(0.0, stump_y), 2, (245, 250, 255), -1, cv2.LINE_AA)

    # Scale bar (10 m).
    bar = int(round(10.0 * scale))
    bx, by = 16, height - 14
    cv2.line(panel, (bx, by), (bx + bar, by), (220, 230, 240), 2, cv2.LINE_AA)
    draw_text(panel, "10 m", (bx + bar + 6, by + 4), scale=0.34, color=(210, 220, 232))

    live = sorted(frame_grounds)
    for player_id in live:
        x, y = frame_grounds[player_id]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pixel = w2p(float(x), float(y))
        trail = trails.setdefault(str(player_id), [])
        trail.append(pixel)
        if len(trail) > trail_len:
            del trail[:-trail_len]
        color = color_for_global_id(str(player_id))
        if len(trail) >= 2:
            cv2.polylines(panel, [np.asarray(trail, np.int32)], False, color, 1, cv2.LINE_AA)
        single = single_ids is not None and str(player_id) in single_ids
        cv2.circle(panel, pixel, 7, (8, 14, 10), -1, cv2.LINE_AA)          # dark halo
        if single:
            cv2.circle(panel, pixel, 5, color, 2, cv2.LINE_AA)             # hollow = 1 camera
        else:
            cv2.circle(panel, pixel, 5, color, -1, cv2.LINE_AA)
            cv2.circle(panel, pixel, 5, (245, 248, 255), 1, cv2.LINE_AA)   # white keyline
        _bev_label(panel, str(player_id), pixel, color)
    # Ghost (disappeared) ids: greyed marker at last-known position, no trail growth.
    for player_id, (x, y) in sorted((ghost_grounds or {}).items()):
        if not (np.isfinite(x) and np.isfinite(y)) or str(player_id) in frame_grounds:
            continue
        pixel = w2p(float(x), float(y))
        base = color_for_global_id(str(player_id))
        grey = tuple(int((c + 150) // 2) for c in base)
        cv2.circle(panel, pixel, 6, (8, 14, 10), -1, cv2.LINE_AA)
        cv2.circle(panel, pixel, 5, grey, 1, cv2.LINE_AA)
        cv2.drawMarker(panel, pixel, grey, cv2.MARKER_TILTED_CROSS, 8, 1, cv2.LINE_AA)
        _bev_label(panel, str(player_id), pixel, grey)

    # Header bar.
    blend_rect(panel, (0, 0), (width, header_h), (12, 20, 16), 0.7)
    cv2.line(panel, (0, 0), (width, 0), (82, 220, 255), 3, cv2.LINE_AA)
    draw_text(panel, "BIRD'S-EYE VIEW", (14, 23), scale=0.56, color=(250, 252, 255), thickness=2)
    ghost_note = f"  ghost {len(ghost_grounds)}" if ghost_grounds else ""
    draw_text(panel, f"f{frame_index}  live {len(live)}{ghost_note}",
              (width - 168, 23), scale=0.42, color=(206, 216, 232))
    return panel


def _bev_label(panel: np.ndarray, text: str, pixel: tuple[int, int], color) -> None:
    """Small readable id label beside a bird's-eye marker (dark keyline for contrast)."""

    x, y = pixel[0] + 8, pixel[1] - 6
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (8, 12, 10), 3, cv2.LINE_AA)
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)


def compute_ground_extents(
    ground_positions: dict[int, dict[str, tuple[float, float]]],
    *,
    margin_m: float = 8.0,
) -> tuple[float, float, float, float]:
    xs, ys = [], []
    for frame in ground_positions.values():
        for x, y in frame.values():
            if np.isfinite(x) and np.isfinite(y):
                xs.append(float(x)); ys.append(float(y))
    if not xs:
        return -50.0, 50.0, -50.0, 50.0
    return min(xs) - margin_m, max(xs) + margin_m, min(ys) - margin_m, max(ys) + margin_m


@functools.lru_cache(maxsize=1)
def _gpu_jpeg_decoder():
    """Return (torch, decode_jpeg, read_file) if GPU JPEG decode is usable, else None.

    Offloads the dominant per-frame CPU cost (entropy-decoding 7 large JPEGs) to the
    GPU's nvJPEG engine, leaving the CPU only the drawing/compositing. Set
    ``QT_RENDER_GPU_DECODE=0`` to force the CPU path.
    """
    import os
    if os.environ.get("QT_RENDER_GPU_DECODE", "1") == "0":
        return None
    try:
        import torch
        from torchvision.io import decode_jpeg, read_file
        if not torch.cuda.is_available():
            return None
        return (torch, decode_jpeg, read_file)
    except Exception:
        return None


def load_image_for_record(camera_dir: Path, record: dict[str, Any]) -> np.ndarray:
    image_path = camera_dir / record["frame_name"]
    gpu = _gpu_jpeg_decoder()
    if gpu is not None and str(image_path).lower().endswith((".jpg", ".jpeg")):
        try:
            torch, decode_jpeg, read_file = gpu
            data = read_file(str(image_path))
            rgb_chw = decode_jpeg(data, device="cuda")           # (3, H, W) RGB uint8 on GPU
            bgr_hwc = rgb_chw.flip(0).permute(1, 2, 0).contiguous()  # -> BGR, HWC (cv2 layout)
            return bgr_hwc.cpu().numpy()
        except Exception:
            pass  # fall back to CPU decode on any GPU/codec hiccup
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    return image


def render_per_camera_videos(
    *,
    records_by_camera: dict[str, list[dict[str, Any]]],
    camera_dirs: dict[str, Path],
    output_dir: Path,
    settings: VideoSettings,
    ball_positions: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    width, height = settings.per_camera_size
    for camera_id, records in records_by_camera.items():
        output_path = output_dir / f"{records[0]['delivery_id']}__{camera_id}.mp4"
        progress = tqdm(records, desc=f"video {camera_id}", unit="frame") if tqdm else records
        with VideoSink(
            output_path,
            width=width,
            height=height,
            fps=settings.fps,
            crf=settings.crf,
            preset=settings.preset,
            use_ffmpeg=settings.use_ffmpeg,
        ) as sink:
            for index, record in enumerate(progress):
                image = load_image_for_record(camera_dirs[camera_id], record)
                frame = render_feed_frame(
                    image,
                    record,
                    output_size=(width, height),
                    keypoint_threshold=settings.keypoint_threshold,
                    compact=False,
                    ball_trail=ball_trail_for(
                        records, index, ball_positions, settings.ball_trail_frames
                    ),
                    show=settings.show,
                )
                sink.write(frame)
        written.append(
            {
                "camera_id": camera_id,
                "path": str(output_path),
                "frames": len(records),
                "size": [width, height],
                "fps": settings.fps,
            }
        )
    return written


def fallback_mosaic_layout(camera_ids: Iterable[str]) -> MosaicLayout:
    """Alphabetical grid used when calibration is unavailable."""

    cameras = sorted(camera_ids)
    slots: list[str | None] = list(cameras[:6]) + [None] * max(0, 6 - len(cameras))
    bottom: list[str | None] = [MONITOR_SLOT, cameras[6] if len(cameras) > 6 else None, ROSTER_SLOT]
    return MosaicLayout(
        grid=(tuple(slots[0:3]), tuple(slots[3:6]), tuple(bottom)),
        mirrored=frozenset(),
        bowling_direction_xy=None,
        notes=("calibration unavailable - alphabetical fallback layout",),
    )


def render_mosaic_video(
    *,
    records_by_camera: dict[str, list[dict[str, Any]]],
    camera_dirs: dict[str, Path],
    output_path: Path,
    settings: VideoSettings,
    run_id: str,
    delivery_id: str,
    ball_positions: dict[str, tuple[float, float]],
    layout: MosaicLayout,
    projections: dict[str, np.ndarray] | None = None,
    ground_positions: dict[int, dict[str, tuple[float, float]]] | None = None,
    ghost_window_frames: int = 150,
    ghost_decay_frames: int = 100,
    roles: dict[str, str] | None = None,
    suppressed: set[str] | None = None,
) -> dict[str, Any]:
    width, height = settings.mosaic_size
    cell_width = width // 3
    cell_height = height // 3
    record_maps = {
        camera_id: {int(record["frame_index"]): (index, record) for index, record in enumerate(records)}
        for camera_id, records in records_by_camera.items()
    }
    common_frames = sorted(set.intersection(*(set(items) for items in record_maps.values())))
    frame_count = len(common_frames)
    roster = build_delivery_roster(records_by_camera, roles)
    projections = projections or {}
    ground_positions = ground_positions or {}
    # Bird's-eye tile replaces the text delivery monitor: fixed world window + per-id trails.
    bev_extents = compute_ground_extents(ground_positions)
    bev_trails: dict[str, list[tuple[int, int]]] = {}
    # Last-known fused ground position per global id, updated every frame it is
    # placed. Lets a ghost be drawn for an id that has vanished from EVERY camera
    # (its fused position is no longer emitted) for up to ``ghost_decay_frames``.
    last_known_ground: dict[str, tuple[tuple[float, float], int]] = {}

    # Per (camera, id) detection frames — used to decide when a missing player is
    # "occluded HERE" (seen in this camera nearby in time) vs simply out of view.
    import bisect

    id_frames: dict[tuple[str, str], list[int]] = {}
    for camera_id, records in records_by_camera.items():
        for record in records:
            frame_index = int(record["frame_index"])
            for player in record.get("players", []):
                player_id = player.get("global_player_id")
                if player_id:
                    id_frames.setdefault((camera_id, str(player_id)), []).append(frame_index)
    for frames in id_frames.values():
        frames.sort()

    def seen_here_nearby(camera_id: str, player_id: str, frame_index: int) -> bool:
        frames = id_frames.get((camera_id, player_id))
        if not frames:
            return False
        pos = bisect.bisect_left(frames, frame_index)
        for neighbour in (pos - 1, pos):
            if 0 <= neighbour < len(frames) and abs(frames[neighbour] - frame_index) <= ghost_window_frames:
                return True
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with VideoSink(
        output_path,
        width=width,
        height=height,
        fps=settings.fps,
        crf=settings.crf,
        preset=settings.preset,
        use_ffmpeg=settings.use_ffmpeg,
    ) as sink:
        progress = tqdm(range(frame_count), desc="video mosaic", unit="frame") if tqdm else range(frame_count)
        for index in progress:
            synchronized_frame = common_frames[index]
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
            canvas[:] = (10, 14, 21)
            total_players = 0
            visible_now: dict[str, list[str]] = {}
            monitor_slot_position: tuple[int, int] | None = None
            roster_slot_position: tuple[int, int] | None = None
            # Who is detected where THIS frame (for occlusion ghosts).
            detected_by_camera: dict[str, set[str]] = {}
            for camera_id in records_by_camera:
                _, record = record_maps[camera_id][synchronized_frame]
                detected_by_camera[camera_id] = {
                    str(player["global_player_id"])
                    for player in record.get("players", [])
                    if player.get("global_player_id")
                }
            frame_grounds = ground_positions.get(synchronized_frame, {})
            for player_id, xy in frame_grounds.items():
                if np.isfinite(xy[0]) and np.isfinite(xy[1]):
                    last_known_ground[str(player_id)] = (
                        (float(xy[0]), float(xy[1])), synchronized_frame,
                    )
            # Ids placed on the pitch but detected by NO camera this frame: draw them
            # greyed in the BEV so a full disappearance is visible where it happens.
            detected_anywhere = set().union(*detected_by_camera.values()) if detected_by_camera else set()
            ghost_grounds = {
                pid: pos
                for pid, (pos, seen) in last_known_ground.items()
                if pid not in detected_anywhere
                and pid not in frame_grounds
                and synchronized_frame - seen <= ghost_decay_frames
            }
            for row, slots in enumerate(layout.grid):
                for col, slot in enumerate(slots):
                    if slot == MONITOR_SLOT:
                        monitor_slot_position = (row, col)
                        continue
                    if slot == ROSTER_SLOT:
                        roster_slot_position = (row, col)
                        continue
                    if slot is None or slot not in records_by_camera:
                        continue
                    camera_id = slot
                    record_index, record = record_maps[camera_id][synchronized_frame]
                    image = load_image_for_record(camera_dirs[camera_id], record)
                    ghosts: list[Ghost] = []
                    projection = projections.get(camera_id)
                    if projection is not None and settings.show == "p4":
                        image_wh = (image.shape[1], image.shape[0])
                        detected_here = detected_by_camera.get(camera_id, set())
                        others_see = set().union(*(
                            ids for cam, ids in detected_by_camera.items() if cam != camera_id
                        )) if len(detected_by_camera) > 1 else set()
                        # A ghost is drawn for any id NOT detected in THIS camera whose
                        # (fused or last-known) ground point is geometrically visible here.
                        # "occluded" = another camera still sees it (current position);
                        # "lost" = no camera sees it this frame (last-known, faded by age).
                        for player_id, (pos, seen_frame) in last_known_ground.items():
                            if player_id in detected_here:
                                continue
                            age = synchronized_frame - seen_frame
                            if age > ghost_decay_frames:
                                continue
                            gx, gy = pos
                            if not ground_point_visible_in(projection, (gx, gy), image_wh):
                                continue
                            foot = project_ground_to_pixel(projection, (gx, gy))
                            if not np.isfinite(foot).all():
                                continue
                            occluded = player_id in others_see
                            if not occluded and not seen_here_nearby(
                                camera_id, player_id, synchronized_frame
                            ):
                                # never seen here recently and gone everywhere: still
                                # show it (that is the point), but only within decay.
                                pass
                            alpha = 1.0 if occluded else max(0.2, 1.0 - age / max(ghost_decay_frames, 1))
                            ghosts.append(Ghost(
                                player_id, (float(foot[0]), float(foot[1])),
                                "occluded" if occluded else "lost", alpha,
                            ))
                    tile = render_feed_frame(
                        image,
                        record,
                        output_size=(cell_width, cell_height),
                        keypoint_threshold=settings.keypoint_threshold,
                        compact=True,
                        ball_trail=ball_trail_for(
                            records_by_camera[camera_id], record_index, ball_positions,
                            settings.ball_trail_frames,
                        ),
                        show=settings.show,
                        mirror=camera_id in layout.mirrored,
                        ghosts=ghosts,
                        letterbox=settings.letterbox_tiles,
                        roles=roles,
                        suppressed=suppressed,
                        body_paint=settings.body_paint,
                    )
                    total_players += len(record.get("players", []))
                    for player in record.get("players", []):
                        player_id = player.get("global_player_id")
                        if player_id:
                            visible_now.setdefault(str(player_id), []).append(camera_id)
                    y1, x1 = row * cell_height, col * cell_width
                    canvas[y1 : y1 + cell_height, x1 : x1 + cell_width] = tile

            direction_note = "unknown"
            if layout.bowling_direction_xy is not None:
                dx, dy = layout.bowling_direction_xy
                direction_note = f"({dx:+.2f}, {dy:+.2f})"
            # An id detected in exactly one camera this frame is single-camera-localised
            # (no cross-camera correction) -> drawn hollow so a lone-camera placement is
            # visible at a glance.
            id_cam_counts: dict[str, int] = {}
            for ids in detected_by_camera.values():
                for pid in ids:
                    id_cam_counts[pid] = id_cam_counts.get(pid, 0) + 1
            single_ids = {pid for pid, n in id_cam_counts.items() if n <= 1}
            monitor = draw_bev_panel(
                size=(cell_width, cell_height),
                frame_grounds={str(pid): xy for pid, xy in frame_grounds.items()},
                extents=bev_extents,
                frame_index=synchronized_frame,
                trails=bev_trails,
                single_ids=single_ids,
                ghost_grounds={str(pid): xy for pid, xy in ghost_grounds.items()},
            )
            roster_panel = draw_roster_panel(
                size=(cell_width, cell_height),
                roster=roster,
                visible_now=visible_now,
            )
            if monitor_slot_position is not None:
                row, col = monitor_slot_position
                canvas[row * cell_height : (row + 1) * cell_height,
                       col * cell_width : (col + 1) * cell_width] = monitor
            if roster_slot_position is not None:
                row, col = roster_slot_position
                canvas[row * cell_height : (row + 1) * cell_height,
                       col * cell_width : (col + 1) * cell_width] = roster_panel
            sink.write(canvas)

    return {
        "path": str(output_path),
        "frames": frame_count,
        "size": [width, height],
        "fps": settings.fps,
        "layout": layout.describe(),
        "layout_notes": list(layout.notes),
    }


def render_ground_tracks(
    *,
    ground_rows: list[dict[str, Any]],
    output_path: Path,
    settings: VideoSettings,
    pitch_extents: tuple[float, float, float, float],
    delivery_id: str,
) -> dict[str, Any]:
    """Render a top-down, identity-coloured world-ground trajectory video."""

    width, height = settings.per_camera_size
    margin = 64
    x_min, x_max, y_min, y_max = pitch_extents

    def world_to_pixel(point: Iterable[float]) -> tuple[int, int]:
        x, y = [float(value) for value in point]
        px = margin + (x - x_min) / max(x_max - x_min, 1e-6) * (width - 2 * margin)
        py = height - margin - (y - y_min) / max(y_max - y_min, 1e-6) * (height - 2 * margin)
        return int(round(px)), int(round(py))

    sampled = ground_rows[:: settings.sample_every]
    if settings.max_frames is not None:
        sampled = sampled[: settings.max_frames]
    trails: dict[str, list[tuple[int, int]]] = {}
    with VideoSink(
        output_path,
        width=width,
        height=height,
        fps=settings.fps,
        crf=settings.crf,
        preset=settings.preset,
        use_ffmpeg=settings.use_ffmpeg,
    ) as sink:
        progress = tqdm(sampled, desc="ground tracks", unit="frame") if tqdm else sampled
        for row in progress:
            canvas = np.full((height, width, 3), (14, 45, 25), dtype=np.uint8)
            cv2.rectangle(canvas, (margin, margin), (width - margin, height - margin), (95, 170, 105), 2)
            cv2.line(canvas, world_to_pixel((x_min, 0.0)), world_to_pixel((x_max, 0.0)), (90, 130, 95), 1)
            # Regulation pitch is approximately 3.05 x 20.12 metres around origin.
            cv2.rectangle(canvas, world_to_pixel((-1.525, 10.06)), world_to_pixel((1.525, -10.06)), (185, 205, 190), 2)
            active = []
            for track in row.get("tracks", []):
                player_id = str(track["global_player_id"])
                pixel = world_to_pixel(track["ground_xy"])
                trail = trails.setdefault(player_id, [])
                trail.append(pixel)
                if len(trail) > 150:
                    del trail[:-150]
                color = color_for_global_id(player_id)
                if len(trail) >= 2:
                    cv2.polylines(canvas, [np.asarray(trail, np.int32)], False, color, 2, cv2.LINE_AA)
                cv2.circle(canvas, pixel, 7, (10, 18, 13), -1, cv2.LINE_AA)
                cv2.circle(canvas, pixel, 5, color, -1, cv2.LINE_AA)
                draw_text(canvas, player_id, (pixel[0] + 8, pixel[1] - 7), scale=0.45, color=color)
                active.append(player_id)
            add_header(
                canvas,
                title=f"GROUND TRACKS  |  {delivery_id}",
                subtitle="calibrated pitch coordinates (metres)",
                right_text=f"frame {row['frame_index']}  active {len(active)}",
                compact=False,
            )
            y = 92
            for player_id in sorted(active)[:12]:
                cv2.circle(canvas, (width - 180, y - 5), 5, color_for_global_id(player_id), -1)
                draw_text(canvas, player_id, (width - 165, y), scale=0.42)
                y += 24
            sink.write(canvas)
    return {
        "path": str(output_path),
        "frames": len(sampled),
        "size": [width, height],
        "fps": settings.fps,
        "pitch_extents_world_m": list(pitch_extents),
    }


def derive_layout_for_run(
    records_by_camera: dict[str, list[dict[str, Any]]],
    drive_root: Path,
    delivery_id: str,
    bowling_end_cam: str | None,
) -> MosaicLayout:
    """Calibration-derived mosaic layout, falling back gracefully without it."""

    try:
        from core.calibration import (
            build_ground_calibrators,
            current_calibration_dir,
            load_projection_matrices_from_drive,
        )
        from identity.p2_tracking.runner import infer_match_id

        match_id = infer_match_id(delivery_id)
        projections = {
            camera_id: matrix
            for camera_id, matrix in load_projection_matrices_from_drive(
                drive_root, match_id
            ).items()
            if camera_id in records_by_camera
        }
        direction = None
        if bowling_end_cam is None:
            axis = load_pitch_axis(
                current_calibration_dir(drive_root, match_id) / "pitch_calibration_config.json"
            )
            if axis is not None:
                calibrators = build_ground_calibrators(
                    drive_root, match_id, sorted(records_by_camera)
                )
                series: dict[str, list[tuple[int, Any]]] = {}
                for camera_id, records in records_by_camera.items():
                    calibrator = calibrators.get(camera_id)
                    if calibrator is None:
                        continue
                    for record in records:
                        for player in record.get("players", []):
                            local_track_id = player.get("local_track_id")
                            bbox = player.get("bbox_xywh_px")
                            if not local_track_id or not bbox:
                                continue
                            xy = calibrator.bbox_bottom_center_ground_xy(
                                [float(value) for value in bbox]
                            )
                            if xy is not None:
                                series.setdefault(f"{camera_id}:{local_track_id}", []).append(
                                    (int(record["frame_index"]), xy)
                                )
                direction = infer_bowling_direction(series, axis)
        return derive_mosaic_layout(
            projections,
            bowling_direction_xy=direction,
            bowling_end_cam=bowling_end_cam,
        )
    except Exception as exc:  # calibration missing/unreadable: degrade, don't die
        layout = fallback_mosaic_layout(records_by_camera)
        return MosaicLayout(
            grid=layout.grid,
            mirrored=layout.mirrored,
            bowling_direction_xy=None,
            notes=layout.notes + (f"layout derivation failed: {exc}",),
        )


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    drive_root = Path(args.drive_root)
    if not drive_root.is_absolute():
        drive_root = ROOT / drive_root

    manifest = json.loads((run_dir / "run_manifest.json").read_text())
    run_id = manifest.get("run_id") or run_dir.name
    show = args.show or stage_from_manifest(manifest)

    prediction_dir = run_dir / "predictions"
    prediction_parts = [
        parts
        for path in sorted(prediction_dir.glob("*.jsonl"))
        if (parts := parse_prediction_filename(path)) is not None
    ]
    if not prediction_parts:
        raise RuntimeError(f"no canonical prediction JSONL files under {prediction_dir}")
    available_deliveries = sorted({delivery for _, delivery, _ in prediction_parts})

    delivery_id = args.delivery_id
    if delivery_id is None:
        # Fall back to the manifest, then to the lone delivery in the run.
        delivery_id = manifest.get("delivery_id")
    if delivery_id is None:
        if len(available_deliveries) != 1:
            raise RuntimeError(
                "run holds multiple deliveries "
                f"({', '.join(available_deliveries)}); pass --delivery-id to choose one"
            )
        delivery_id = available_deliveries[0]
    if delivery_id not in available_deliveries:
        raise RuntimeError(
            f"delivery {delivery_id!r} not found in {prediction_dir}; "
            f"available: {', '.join(available_deliveries)}"
        )

    # Map cam_NN -> prediction JSONL for the chosen delivery.
    prediction_paths_by_camera = {
        camera_id: prediction_dir / f"{group}__{delivery}__{camera_id}.jsonl"
        for group, delivery, camera_id in prediction_parts
        if delivery == delivery_id
    }

    artifact_dir = Path(args.artifact_dir) if args.artifact_dir else run_dir / "visualizations" / "videos"
    if not artifact_dir.is_absolute():
        artifact_dir = ROOT / artifact_dir

    settings = VideoSettings(
        fps=args.fps,
        keypoint_threshold=args.keypoint_threshold,
        per_camera_size=parse_size(args.per_camera_size),
        mosaic_size=parse_size(args.mosaic_size),
        sample_every=max(1, args.sample_every),
        max_frames=args.max_frames,
        crf=args.crf,
        preset=args.preset,
        use_ffmpeg=args.use_ffmpeg,
        ball_trail_frames=max(0, args.ball_trail_frames),
        show=show,
        letterbox_tiles=args.letterbox_tiles,
        body_paint=args.body_paint,
    )

    selected_cameras = args.cameras or sorted(prediction_paths_by_camera)
    camera_dirs = resolve_delivery_camera_dirs(drive_root, delivery_id)
    ball_positions = load_ball_positions(drive_root / "dataset" / "events-data", delivery_id)
    records_by_camera: dict[str, list[dict[str, Any]]] = {}
    cluster_badges = load_cluster_badges(run_dir / "diagnostics" / "correspondences.jsonl")
    for camera_id in selected_cameras:
        prediction_path = prediction_paths_by_camera.get(camera_id)
        if prediction_path is None:
            if args.cameras is not None:
                raise RuntimeError(f"no prediction JSONL for {camera_id} in delivery {delivery_id}")
            continue
        if camera_id not in camera_dirs:
            raise RuntimeError(f"camera directory not found for {camera_id}")
        if not prediction_path.exists():
            raise RuntimeError(f"prediction file not found: {prediction_path}")
        records = load_records(
            prediction_path,
            sample_every=settings.sample_every,
            max_frames=settings.max_frames,
        )
        if not records:
            raise RuntimeError(f"no records loaded from {prediction_path}")
        for record in records:
            badges = cluster_badges.get((int(record["frame_index"]), camera_id), {})
            for player_index, player in enumerate(record.get("players", [])):
                if player_index in badges:
                    player["_cluster_id"] = badges[player_index]
        records_by_camera[camera_id] = records

    outputs: dict[str, Any] = {
        "schema_version": "cricket_pipetrack_video_manifest/v2",
        "run_id": run_id,
        "delivery_id": delivery_id,
        "artifact_dir": repo_relative(artifact_dir, ROOT),
        "settings": {
            "fps": settings.fps,
            "keypoint_threshold": settings.keypoint_threshold,
            "per_camera_size": list(settings.per_camera_size),
            "mosaic_size": list(settings.mosaic_size),
            "sample_every": settings.sample_every,
            "max_frames": settings.max_frames,
            "encoder": (
                ("ffmpeg/h264_nvenc" if _ffmpeg_has_nvenc() else "ffmpeg/libx264")
                if settings.use_ffmpeg and shutil.which("ffmpeg") else "opencv/mp4v"
            ),
            "crf": settings.crf,
            "preset": settings.preset,
            "show": show,
        },
    }

    if args.mode in {"all", "per-camera"}:
        outputs["per_camera"] = render_per_camera_videos(
            records_by_camera=records_by_camera,
            camera_dirs=camera_dirs,
            output_dir=artifact_dir / "per_camera",
            settings=settings,
            ball_positions=ball_positions,
        )
    if args.mode in {"all", "mosaic"}:
        if len(records_by_camera) < 2:
            raise RuntimeError("mosaic mode requires at least two cameras")
        layout = derive_layout_for_run(
            records_by_camera, drive_root, delivery_id, args.bowling_end_cam
        )
        print(f"mosaic layout: {layout.describe()}")
        for note in layout.notes:
            print(f"  layout note: {note}")
        mosaic_projections: dict[str, np.ndarray] = {}
        try:
            from core.calibration import load_projection_matrices_from_drive
            from identity.p2_tracking.runner import infer_match_id

            mosaic_projections = load_projection_matrices_from_drive(
                drive_root, infer_match_id(delivery_id)
            )
        except Exception:
            pass  # ghosts are an enhancement; the mosaic renders without them
        ground_positions: dict[int, dict[str, tuple[float, float]]] = {}
        ground_tracks_path = run_dir / "diagnostics" / "ground_tracks.jsonl"
        if ground_tracks_path.exists():
            for row in iter_jsonl(ground_tracks_path):
                frame_grounds = {}
                for track in row.get("tracks", []):
                    xy = track.get("ground_xy")
                    player_id = track.get("global_player_id")
                    if player_id and xy and len(xy) >= 2:
                        frame_grounds[str(player_id)] = (float(xy[0]), float(xy[1]))
                if frame_grounds:
                    ground_positions[int(row["frame_index"])] = frame_grounds
        if args.roles_path:
            roles = load_roles(Path(args.roles_path))
        else:
            roles = load_roles(run_dir.parent / "06_roles" / "roles.json") or load_roles(
                run_dir / "06_roles" / "roles.json"
            )
        if roles:
            print("roles loaded: " + ", ".join(f"{pid}={role}" for pid, role in sorted(roles.items())))
        # Wave-6: role-aware peripheral suppression (06 roles); absent/disabled => empty set
        suppressed = load_suppression(run_dir.parent / "06_roles" / "suppression.json") | load_suppression(
            run_dir / "06_roles" / "suppression.json"
        )
        if suppressed:
            print("suppressed ids (P5b): " + ", ".join(sorted(suppressed)))
        outputs["mosaic"] = render_mosaic_video(
            records_by_camera=records_by_camera,
            camera_dirs=camera_dirs,
            output_path=artifact_dir / f"{delivery_id}__all_cameras.mp4",
            settings=settings,
            run_id=run_id,
            delivery_id=delivery_id,
            ball_positions=ball_positions,
            layout=layout,
            projections=mosaic_projections,
            ground_positions=ground_positions,
            roles=roles,
            suppressed=suppressed,
        )
    if args.mode in {"all", "ground"}:
        ground_path = run_dir / "diagnostics" / "ground_tracks.jsonl"
        if ground_path.exists():
            ground_rows = list(iter_jsonl(ground_path))
            outputs["ground_tracks"] = render_ground_tracks(
                ground_rows=ground_rows,
                output_path=artifact_dir / f"{delivery_id}__ground_tracks.mp4",
                settings=settings,
                pitch_extents=load_pitch_extents(
                    drive_root
                    / "dataset"
                    / "calibration-data"
                    / manifest.get("match_id", "CCPL080626")
                    / "calibration_data"
                    / "pitch_calibration_config.json"
                ),
                delivery_id=delivery_id,
            )
        elif args.mode == "ground":
            raise RuntimeError(f"ground mode requires {ground_path}")

    # Store output video paths repo-root-relative for portable manifests.
    for entry in outputs.get("per_camera", []):
        entry["path"] = repo_relative(entry["path"], ROOT)
    if "mosaic" in outputs:
        outputs["mosaic"]["path"] = repo_relative(outputs["mosaic"]["path"], ROOT)
    if "ground_tracks" in outputs:
        outputs["ground_tracks"]["path"] = repo_relative(outputs["ground_tracks"]["path"], ROOT)

    manifest_path = artifact_dir / "video_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote video manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
