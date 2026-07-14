#!/usr/bin/env python3
"""Bird's-eye-view (QT-style) render of player 3D ground positions.

Draws a top-down plot of the cricket ground with every detected player's world
``(x, y)`` position as an ID-coloured dot, animated over a delivery. This is the
instrument for *seeing* 3D-location quality (and, later, identity): single-camera
positions -- which the numeric proxies cannot judge -- are drawn as hollow rings so
you can tell at a glance whether a lone camera is placing a player somewhere sane.

Reads a canonical P4 run dir:
  <p4>/diagnostics/ground_tracks.jsonl   (per-frame global_player_id -> ground_xy)
  <p4>/diagnostics/correspondences.jsonl (per-frame single_camera flag per cluster)

Outputs an mp4 (cv2 VideoWriter, no ffmpeg needed) and a sampled-frame montage PNG.

Example:
  conda activate cricket-rtmpose-l
  python scripts/visualization/render_bird_eye_view.py \
      --p4-dir benchmarks/runs/pipetrack_v3/deliveries/CCPL080626M1_1_14_1/p4 \
      --drive-root drive --match-id CCPL080626 --out-dir /tmp/bev
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_ground_tracks(p4_dir: Path) -> dict[int, list[tuple[str, np.ndarray]]]:
    path = p4_dir / "diagnostics" / "ground_tracks.jsonl"
    out: dict[int, list[tuple[str, np.ndarray]]] = {}
    for row in read_jsonl(path):
        frame = int(row["frame_index"])
        out[frame] = [
            (str(t["global_player_id"]), np.asarray(t["ground_xy"], dtype=float))
            for t in row.get("tracks", [])
            if t.get("ground_xy") is not None and np.isfinite(t["ground_xy"]).all()
        ]
    return out


def load_single_camera_ids(p4_dir: Path) -> dict[int, set[str]]:
    """Per frame, the set of global ids whose cluster was single-camera (best-effort)."""
    path = p4_dir / "diagnostics" / "correspondences.jsonl"
    out: dict[int, set[str]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        frame = int(row["frame_index"])
        singles: set[str] = set()
        for cluster in row.get("clusters", []):
            if cluster.get("single_camera") and cluster.get("binding_id"):
                singles.add(str(cluster["binding_id"]))
        out[frame] = singles
    return out


def load_pitch_points(drive_root: Path, match_id: str) -> np.ndarray:
    path = drive_root / "dataset" / "calibration-data" / match_id / "calibration_data" / "pitch_calibration_config.json"
    pts = []
    if path.exists():
        data = json.load(path.open("r", encoding="utf-8"))
        for key, value in data.items():
            if isinstance(value, list) and len(value) == 3:
                try:
                    pts.append([float(value[0]), float(value[1])])
                except (TypeError, ValueError):
                    continue
    return np.asarray(pts, dtype=float) if pts else np.zeros((0, 2))


def stable_color(global_id: str):
    """Unified id colour (shared with the mosaic via ``identity_colors``).

    ``color_for_global_id`` returns a BGR 0-255 tuple; convert to matplotlib RGB 0-1
    so the standalone bird's-eye view and the in-mosaic BEV/tiles colour every id
    identically instead of the old independent golden-ratio hash.
    """

    from identity.visualization.identity_colors import color_for_global_id

    b, g, r = color_for_global_id(str(global_id))
    return (r / 255.0, g / 255.0, b / 255.0)


def compute_extent(tracks: dict, pitch: np.ndarray) -> tuple[float, float, float, float]:
    xs = [p[0] for frame in tracks.values() for _id, p in frame]
    ys = [p[1] for frame in tracks.values() for _id, p in frame]
    if pitch.size:
        xs += list(pitch[:, 0]); ys += list(pitch[:, 1])
    if not xs:
        return -50, 50, -50, 50
    m = 8.0
    return min(xs) - m, max(xs) + m, min(ys) - m, max(ys) + m


def draw_frame(ax, frame_idx, players, singles, pitch, extent):
    ax.clear()
    x0, x1, y0, y1 = extent
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.grid(True, color="0.9", linewidth=0.5)
    ax.set_facecolor("#0b5c2e")  # grass
    # boundary ellipse (nominal, fit to extent)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ax.add_patch(matplotlib.patches.Ellipse((cx, cy), (x1 - x0) * 0.98, (y1 - y0) * 0.98,
                                            fill=False, edgecolor="white", linewidth=1.5, alpha=0.6))
    if pitch.size:
        ax.scatter(pitch[:, 0], pitch[:, 1], s=3, c="#d9c9a3", alpha=0.7, zorder=2)
    # ``singles`` (from correspondence ``binding_id``) is a DIFFERENT id namespace
    # than the ground-track ``global_player_id`` keyed here, so the old hollow-marker
    # join never matched. Render solid id-coloured dots; the in-mosaic BEV is the tool
    # that distinguishes single-camera (hollow) using the per-frame camera counts.
    del singles
    for gid, xy in players:
        col = stable_color(gid)
        ax.scatter([xy[0]], [xy[1]], s=90, facecolors=col,
                   edgecolors="white", linewidths=1.2, zorder=4)
        ax.annotate(gid, (xy[0], xy[1]), fontsize=7, color="white", zorder=5,
                    xytext=(3, 3), textcoords="offset points")
    ax.set_title(f"frame {frame_idx}   players={len(players)}",
                 color="white", fontsize=9)
    ax.tick_params(colors="0.7", labelsize=7)


def fig_to_bgr(fig) -> np.ndarray:
    fig.canvas.draw()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    rgba = buf.reshape(h, w, 4)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--p4-dir", required=True)
    ap.add_argument("--drive-root", default="drive")
    ap.add_argument("--match-id", default="CCPL080626")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth frame")
    ap.add_argument("--montage-frames", type=int, default=9)
    args = ap.parse_args()

    p4_dir = Path(args.p4_dir).resolve()
    out_dir = Path(args.out_dir).resolve(); out_dir.mkdir(parents=True, exist_ok=True)
    tracks = load_ground_tracks(p4_dir)
    singles = load_single_camera_ids(p4_dir)
    pitch = load_pitch_points(Path(args.drive_root).resolve(), args.match_id)
    extent = compute_extent(tracks, pitch)
    frames = sorted(tracks)[:: max(1, args.stride)]
    if not frames:
        raise SystemExit(f"no ground-track frames in {p4_dir}")

    name = p4_dir.parent.name
    fig, ax = plt.subplots(figsize=(7, 7), dpi=100)
    fig.patch.set_facecolor("#062a16")

    # mp4
    first = fig_to_bgr(fig)
    vpath = out_dir / f"bev_{name}.mp4"
    writer = cv2.VideoWriter(str(vpath), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (first.shape[1], first.shape[0]))
    for f in frames:
        draw_frame(ax, f, tracks[f], singles.get(f, set()), pitch, extent)
        writer.write(fig_to_bgr(fig))
    writer.release()

    # montage
    sample = frames[:: max(1, len(frames) // args.montage_frames)][: args.montage_frames]
    cols = 3; rows = int(np.ceil(len(sample) / cols))
    mfig, maxes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), dpi=100)
    mfig.patch.set_facecolor("#062a16")
    for i, axm in enumerate(np.atleast_1d(maxes).ravel()):
        if i < len(sample):
            draw_frame(axm, sample[i], tracks[sample[i]], singles.get(sample[i], set()), pitch, extent)
        else:
            axm.axis("off")
    mpath = out_dir / f"bev_{name}_montage.png"
    mfig.tight_layout(); mfig.savefig(mpath, facecolor=mfig.get_facecolor()); plt.close(mfig)

    n_single = sum(len(s) for s in singles.values())
    print(f"frames={len(frames)}  extent={tuple(round(e,1) for e in extent)}  single-cam-dots={n_single}")
    print(f"video:   {vpath}")
    print(f"montage: {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
