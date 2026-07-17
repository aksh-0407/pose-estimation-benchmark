#!/usr/bin/env python3
"""Zoomed before/after 3D footage of each END of the cricket field.

Players cluster around the two batting ends. This splits them into two groups by ground
position (2-means) and renders one MP4 per end, tightly zoomed to that group's players, so
per-joint jitter (incl. the face) is clearly visible. Left panel BEFORE (06_roles), right
panel AFTER (07_refine).

    python3 tools/viz_sides.py --delivery CCPL080626M1_1_14_1 \
        --before data/derived/8_init/pipetrack_v90_stack1f \
        --after  data/derived/8_init/refine_demo \
        --out-dir data/viz/refine_demo --step 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_refine import CONNECTIONS, PALETTE, load_players  # noqa: E402

FACE = {"nose", "left_eye", "right_eye", "left_ear", "right_ear", "head"}


def _median_ground(seq: dict) -> np.ndarray | None:
    pts = [J.get("hip") for J in seq.values() if J.get("hip")]
    if not pts:
        pts = [next(iter(J.values())) for J in seq.values() if J]
    if not pts:
        return None
    return np.median(np.array(pts, dtype=float)[:, :2], axis=0)


def two_means(points: np.ndarray, iters: int = 50):
    """Assign each point to one of two clusters (seeded by the farthest-apart pair)."""
    n = len(points)
    if n <= 2:
        return np.arange(n) % 2
    dist = np.linalg.norm(points[:, None] - points[None], axis=2)
    i, j = np.unravel_index(np.argmax(dist), dist.shape)
    centers = points[[i, j]].astype(float)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = np.linalg.norm(points[:, None] - centers[None], axis=2)
        new = d.argmin(1)
        if np.array_equal(new, labels):
            break
        labels = new
        for k in (0, 1):
            if (labels == k).any():
                centers[k] = points[labels == k].mean(0)
    return labels


def _root(J):
    if J.get("hip"):
        return np.array(J["hip"], dtype=float)
    return np.mean(np.array(list(J.values()), dtype=float), axis=0)


def _draw_tracked(ax, J, title, center, half, color):
    """One player, camera tracking their root - fixed ~2*half window so detail stays big."""
    ax.clear()
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_zlim(-0.2, 2 * half - 0.2)   # ground-anchored vertical window
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.view_init(elev=10, azim=-75)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=11)
    if not J:
        return
    body = np.array([c for n, c in J.items() if n not in FACE], dtype=float)
    face = np.array([c for n, c in J.items() if n in FACE], dtype=float)
    if len(body):
        ax.scatter(body[:, 0], body[:, 1], body[:, 2], s=22, color=color)
    if len(face):
        ax.scatter(face[:, 0], face[:, 1], face[:, 2], s=45, color="#d62728")
    for a, b in CONNECTIONS:
        if a in J and b in J:
            ax.plot([J[a][0], J[b][0]], [J[a][1], J[b][1]], [J[a][2], J[b][2]],
                    color=color, lw=2.4, alpha=0.85)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delivery", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--half", type=float, default=1.25, help="half-window (m); ~2*half is the view size")
    ap.add_argument("--players", nargs="*", default=None, help="restrict to these gids")
    args = ap.parse_args()

    before = load_players(args.before, args.delivery, "06_roles", None)
    after = load_players(args.after, args.delivery, "07_refine", None)
    gids = args.players or sorted(set(before) | set(after))
    colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(sorted(set(before) | set(after)))}

    # Label each player by field end (2-means on ground position) for the filename.
    present = [g for g in gids if after.get(g) or before.get(g)]
    grounds = np.array([_median_ground(after.get(g) or before.get(g)) for g in present])
    ends = dict(zip(present, two_means(grounds))) if len(present) > 1 else {present[0]: 0}

    out_dir = Path(args.out_dir)
    for gid in present:
        seq_a = after.get(gid, {})
        seq_b = before.get(gid, {})
        frames = sorted(set(seq_a) | set(seq_b))[:: args.step]
        if not frames:
            continue
        fig = plt.figure(figsize=(15, 7.5))
        axb = fig.add_subplot(1, 2, 1, projection="3d")
        axa = fig.add_subplot(1, 2, 2, projection="3d")

        def update(i, frames=frames, gid=gid, seq_a=seq_a, seq_b=seq_b):
            f = frames[i]
            Ja, Jb = seq_a.get(f, {}), seq_b.get(f, {})
            center = _root(Ja) if Ja else (_root(Jb) if Jb else np.zeros(3))
            _draw_tracked(axb, Jb, f"BEFORE - {gid} f{f}", center, args.half, colors[gid])
            _draw_tracked(axa, Ja, f"AFTER - {gid} f{f}", center, args.half, colors[gid])

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)
        out = out_dir / args.delivery / f"end{ends[gid]}_{gid}.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        ani.save(str(out), writer=animation.FFMpegWriter(fps=20, bitrate=3500), dpi=85)
        plt.close(fig)
        print(f"wrote {out}  (end {ends[gid]}, {gid}, {len(frames)} frames)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
