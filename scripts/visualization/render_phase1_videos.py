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
from typing import Any, Iterable

import cv2
import numpy as np

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover - tqdm is optional in minimal envs.
    tqdm = None

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pose_estimation.cricket.dataset import (
    parse_prediction_filename,
    repo_relative,
    resolve_delivery_camera_dirs,
)
from pose_estimation.cricket.phase1_outputs import COCO_17_EDGES
from scripts.visualization.identity_colors import (
    IDENTITY_PALETTE,
    color_for_global_id,
    color_for_player,
)


CAMERA_ORDER = [f"cam_{index:02d}" for index in range(1, 8)]
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
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if use_ffmpeg and shutil.which("ffmpeg"):
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-vcodec",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "-s",
                f"{width}x{height}",
                "-r",
                str(fps),
                "-i",
                "-",
                "-an",
                "-vcodec",
                "libx264",
                "-preset",
                preset,
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                str(path),
            ]
            self.process = subprocess.Popen(command, stdin=subprocess.PIPE)
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
    parser.add_argument("--mode", choices=["all", "per-camera", "mosaic", "ground"], default="all")
    parser.add_argument("--show", choices=["p2", "p3", "p4"], default=None)
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


def confidence_color(score: float) -> tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    if score >= 0.75:
        return (111, 255, 151)
    if score >= 0.45:
        return (82, 220, 255)
    return (78, 118, 255)


def draw_players(
    image: np.ndarray,
    record: dict[str, Any],
    *,
    sx: float,
    sy: float,
    keypoint_threshold: float,
    compact: bool,
    show: str,
) -> int:
    players = [
        scale_player(player, sx=sx, sy=sy)
        for player in record.get("players", [])
        if player.get("bbox_xywh_px") is not None and player.get("pose_2d") is not None
    ]
    line_thickness = 2 if compact else 3
    point_radius = 3 if compact else 5
    font_scale = 0.42 if compact else 0.56

    for player_index, player in enumerate(players, start=1):
        if show == "p3" and player.get("_cluster_id") is not None:
            color = color_for_global_id(
                f"cluster:{record['frame_index']}:{player['_cluster_id']}"
            )
        else:
            color = color_for_player(player.get("global_player_id"), player.get("local_track_id"))
        x, y, w, h = [int(round(value)) for value in player["bbox_xywh_px"]]
        x2, y2 = x + w, y + h
        detection_confidence = player.get("detection_confidence")
        if detection_confidence is None:
            detection_confidence = player.get("track_confidence")
        detection_conf = float(detection_confidence or 0.0)

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
        label = f"{identity}  trk {float(player.get('track_confidence') or 0.0):.2f}"
        label_y = max(24, y - 8)
        next_x, _ = draw_chip(image, label, (max(8, x), label_y), color=color, scale=font_scale)
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
) -> np.ndarray:
    output_width, output_height = output_size
    source_height, source_width = image.shape[:2]
    frame = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
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
    )
    if ball_trail:
        draw_ball_trail(frame, ball_trail, compact=compact)
    camera_title = record["camera_id"].replace("_", " ").upper()
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


def load_image_for_record(camera_dir: Path, record: dict[str, Any]) -> np.ndarray:
    image_path = camera_dir / record["frame_name"]
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


def render_mosaic_video(
    *,
    records_by_camera: dict[str, list[dict[str, Any]]],
    camera_dirs: dict[str, Path],
    output_path: Path,
    settings: VideoSettings,
    run_id: str,
    delivery_id: str,
    ball_positions: dict[str, tuple[float, float]],
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
            active_ids: set[str] = set()
            for camera_position, camera_id in enumerate(CAMERA_ORDER):
                if camera_id not in records_by_camera:
                    continue
                row, col = divmod(camera_position, 3)
                record_index, record = record_maps[camera_id][synchronized_frame]
                image = load_image_for_record(camera_dirs[camera_id], record)
                tile = render_feed_frame(
                    image,
                    record,
                    output_size=(cell_width, cell_height),
                    keypoint_threshold=settings.keypoint_threshold,
                    compact=True,
                    ball_trail=ball_trail_for(
                        records_by_camera[camera_id], record_index, ball_positions, settings.ball_trail_frames
                    ),
                    show=settings.show,
                )
                total_players += len(record.get("players", []))
                active_ids.update(
                    str(player["global_player_id"])
                    for player in record.get("players", [])
                    if player.get("global_player_id")
                )
                y1, x1 = row * cell_height, col * cell_width
                canvas[y1 : y1 + cell_height, x1 : x1 + cell_width] = tile

            frame_index = synchronized_frame
            summary = draw_info_panel(
                size=(cell_width, cell_height),
                title="Delivery Monitor",
                lines=[
                    delivery_id,
                    f"run: {run_id}",
                    f"frame: {frame_index}",
                    f"feeds: {len(records_by_camera)} cameras",
                    f"detections: {total_players}",
                    f"joint threshold: {settings.keypoint_threshold:.2f}",
                ],
                accent=(82, 220, 255),
            )
            legend = draw_info_panel(
                size=(cell_width, cell_height),
                title="Active Global Roster",
                lines=(
                    [f"{player_id}  identity-stable colour" for player_id in sorted(active_ids)[:6]]
                    or ["No confirmed global IDs", "P2/P3 fallback uses local IDs"]
                ),
                accent=(129, 236, 145),
            )
            y = 2 * cell_height
            canvas[y : y + cell_height, cell_width : 2 * cell_width] = summary
            canvas[y : y + cell_height, 2 * cell_width : 3 * cell_width] = legend
            sink.write(canvas)

    return {
        "path": str(output_path),
        "frames": frame_count,
        "size": [width, height],
        "fps": settings.fps,
        "layout": "3x3: cam_01..cam_07 plus summary and legend panels",
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
    )

    selected_cameras = args.cameras or CAMERA_ORDER
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
            "encoder": "ffmpeg/libx264" if settings.use_ffmpeg and shutil.which("ffmpeg") else "opencv/mp4v",
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
        if [camera for camera in CAMERA_ORDER if camera in records_by_camera] != CAMERA_ORDER:
            raise RuntimeError("mosaic mode requires all seven cameras")
        outputs["mosaic"] = render_mosaic_video(
            records_by_camera=records_by_camera,
            camera_dirs=camera_dirs,
            output_path=artifact_dir / f"{delivery_id}__all_cameras.mp4",
            settings=settings,
            run_id=run_id,
            delivery_id=delivery_id,
            ball_positions=ball_positions,
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
