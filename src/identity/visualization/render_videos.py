#!/usr/bin/env python3
"""Render pipeline predictions as MP4 videos (per-camera, mosaic, ground tracks).

This is the CLI and orchestration layer. The building blocks live beside it:

- ``video_io``: encoders (``VideoSink``) and frame decoding.
- ``loaders``: JSONL prediction and side-artifact readers.
- ``overlays``: everything drawn on a camera picture (skeletons, chips, ghosts).
- ``panels``: standalone mosaic tiles (roster, bird's-eye, info).
- ``mosaic_layout``: calibration-derived camera arrangement.
"""

from __future__ import annotations

import argparse
import json
import shutil
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

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from core.frames import (  # noqa: E402
    parse_prediction_filename,
    repo_relative,
    resolve_delivery_camera_dirs,
)
from core.datasets import calibration_root_for  # noqa: E402
from identity.common.geometry import (  # noqa: E402
    ground_point_visible_in,
    project_ground_to_pixel,
)
from identity.visualization.identity_colors import (  # noqa: E402
    IDENTITY_PALETTE,
    color_for_global_id,
)
from identity.visualization.loaders import (  # noqa: E402
    ball_trail_for,
    build_delivery_roster,
    compute_ground_extents,
    iter_jsonl,
    load_ball_positions,
    load_cluster_badges,
    load_pitch_extents,
    load_records,
    load_roles,
    load_suppression,
    stage_from_manifest,
)
from identity.visualization.mosaic_layout import (  # noqa: E402
    MONITOR_SLOT,
    ROSTER_SLOT,
    MosaicLayout,
    derive_mosaic_layout,
    infer_bowling_direction,
    load_pitch_axis,
)
from identity.visualization.overlays import (  # noqa: E402
    BALL_TRAIL_FRAMES,
    Ghost,
    _rects_overlap,
    add_header,
    draw_body_paint,
    draw_text,
    place_chip,
    render_feed_frame,
)
from identity.visualization.panels import (  # noqa: E402
    draw_bev_panel,
    draw_roster_panel,
)
from identity.visualization.video_io import (  # noqa: E402
    VideoSink,
    _ffmpeg_has_nvenc,
    load_image_for_record,
    parse_size,
)

PLAYER_COLORS = list(IDENTITY_PALETTE)  # backward-compatible export for callers


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
        help="Roles JSON (default: probes <run-dir>/../06_roles/roles.json then <run-dir>/06_roles/roles.json)",
    )
    parser.add_argument("--mode", choices=["all", "per-camera", "mosaic", "ground"], default="all")
    parser.add_argument("--show", choices=["p2", "p3", "p4"], default=None)
    parser.add_argument("--no-body-paint", dest="body_paint", action="store_false",
                        help="Disable the skeleton body-paint identity overlay (on by default)")
    parser.add_argument("--letterbox-tiles", action="store_true",
                        help="Aspect-correct wide tiles (C07) by bottom-padding instead of stretching.")
    parser.add_argument("--crf", type=int, default=22)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument("--no-ffmpeg", dest="use_ffmpeg", action="store_false")
    parser.set_defaults(use_ffmpeg=True)
    return parser.parse_args()


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
    # Bird's-eye tile: fixed world window plus per-id trails across frames.
    bev_extents = compute_ground_extents(ground_positions)
    bev_trails: dict[str, list[tuple[int, int]]] = {}
    # Last-known fused ground position per global id, updated every frame it is
    # placed. Lets a ghost be drawn for an id that has vanished from EVERY camera
    # (its fused position is no longer emitted) for up to ``ghost_decay_frames``.
    last_known_ground: dict[str, tuple[tuple[float, float], int]] = {}

    # Per (camera, id) detection frames: used to decide when a missing player is
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
            # Ids placed on the pitch but detected by NO camera this frame: drawn
            # greyed in the bird's-eye so a full disappearance is visible where it happens.
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
                                # Never seen here recently and gone everywhere: still
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

            # An id detected in exactly one camera this frame is single-camera
            # localised (no cross-camera correction), drawn hollow so a lone-camera
            # placement is visible at a glance.
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
    # Cap OpenCV's internal thread pool: this renderer runs one process per delivery
    # under the batch driver's render pool, and cv2 otherwise grabs all cores per
    # process. BLAS caps from the driver do NOT cover OpenCV, so cap in-process.
    cv2.setNumThreads(1)
    try:
        cv2.ocl.setUseOpenCL(False)
    except Exception as exc:
        print(f"WARN: could not disable OpenCL ({exc}); continuing", flush=True)
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
    ball_positions = load_ball_positions(drive_root / "events-data", delivery_id)
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
        except Exception as exc:
            # Ghost markers are an enhancement; the mosaic renders without them.
            print(f"WARN: projections unavailable ({exc}); rendering without ghost markers", flush=True)
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
        # Role-aware peripheral suppression (06 roles); absent/disabled => empty set.
        suppressed = load_suppression(run_dir.parent / "06_roles" / "suppression.json") | load_suppression(
            run_dir / "06_roles" / "suppression.json"
        )
        if suppressed:
            print("suppressed ids: " + ", ".join(sorted(suppressed)))
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
                    calibration_root_for(drive_root)
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
