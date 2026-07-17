#!/usr/bin/env python3
"""Headless before/after animation using the company's animation_viz style.

Reuses the shared CONNECTIONS + facing-direction arrow from
``identity.visualization.animation_viz`` but renders offline to an MP4 (two panels:
BEFORE = 06_roles, AFTER = 07_refine) locked to one player, so the refinement can be
reviewed without an interactive window.

    python3 tools/animate_refine.py --delivery CCPL080626M1_1_14_1 --player P002 \
        --before data/derived/8_init/pipetrack_v90_stack1f \
        --after  data/derived/8_init/refine_demo \
        --out data/viz/refine_demo/animation_14_1_P002.mp4 --step 2
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.animation as animation  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from identity.visualization.animation_viz import CONNECTIONS, calculate_facing_direction  # noqa: E402


PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]


def load_players(tree: str, delivery: str, stage: str, player: str | None) -> dict[str, dict[int, dict]]:
    """{gid: {frame_index: {joint_name: (x,y,z)}}} from pose_3d_named; player=None -> all."""
    out: dict[str, dict[int, dict]] = {}
    for path in sorted(glob.glob(f"{tree}/{delivery}/{stage}/predictions/*.jsonl")):
        for line in open(path):
            if not line.strip():
                continue
            rec = json.loads(line)
            frame = int(rec["frame_index"])
            for pl in rec.get("players", []):
                gid = pl.get("global_player_id")
                if not gid or (player and gid != player):
                    continue
                if frame in out.get(gid, {}):
                    continue
                nm = pl.get("pose_3d_named")
                if not nm:
                    continue
                rw = nm["root_world_m"]
                out.setdefault(gid, {})[frame] = {
                    k: (rw[0] + v[0], rw[1] + v[1], rw[2] + v[2])
                    for k, v in nm["joints_root_relative_m"].items() if v
                }
    return out


def _bounds(players: dict[str, dict[int, dict]]):
    pts = np.array([c for seq in players.values() for J in seq.values() for c in J.values()], dtype=float)
    if len(pts) == 0:
        return np.zeros(3), np.ones(3)
    lo, hi = pts.min(0), pts.max(0)
    mid = (lo + hi) / 2
    rng = float((hi - lo).max()) / 2 or 1.0
    return mid - rng, mid + rng


def _draw(ax, players: dict[str, dict[int, dict]], frame: int, title, lo, hi, colors):
    ax.clear()
    ax.set_xlim(lo[0], hi[0]); ax.set_ylim(lo[1], hi[1]); ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(title, fontsize=10)
    for gid, seq in players.items():
        J = seq.get(frame)
        if not J:
            continue
        color = colors[gid]
        xs = [c[0] for c in J.values()]; ys = [c[1] for c in J.values()]; zs = [c[2] for c in J.values()]
        ax.scatter(xs, ys, zs, s=10, color=color)
        for a, b in CONNECTIONS:
            if a in J and b in J:
                ax.plot([J[a][0], J[b][0]], [J[a][1], J[b][1]], [J[a][2], J[b][2]],
                        color=color, lw=1.6, alpha=0.75)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--delivery", required=True)
    ap.add_argument("--player", default="ALL", help="global_player_id, or ALL for the whole scene")
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=3)
    args = ap.parse_args()

    target = None if args.player == "ALL" else args.player
    before = load_players(args.before, args.delivery, "06_roles", target)
    after = load_players(args.after, args.delivery, "07_refine", target)
    gids = sorted(set(before) | set(after))
    colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(gids)}
    frames = sorted(set().union(*[set(before.get(g, {})) for g in gids],
                                *[set(after.get(g, {})) for g in gids]))[:: args.step]
    if not frames:
        print(f"no frames for {args.delivery}")
        return 1
    lo_b, hi_b = _bounds(before)
    lo_a, hi_a = _bounds(after)

    fig = plt.figure(figsize=(15, 7))
    axb = fig.add_subplot(1, 2, 1, projection="3d")
    axa = fig.add_subplot(1, 2, 2, projection="3d")

    def update(i):
        f = frames[i]
        _draw(axb, before, f, f"BEFORE (06_roles) - {args.delivery} f{f}", lo_b, hi_b, colors)
        _draw(axa, after, f, f"AFTER (07_refine) - f{f}", lo_a, hi_a, colors)

    ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = animation.FFMpegWriter(fps=20, bitrate=3000)
    ani.save(args.out, writer=writer, dpi=80)
    print(f"wrote {args.out}  ({len(frames)} frames, {len(gids)} players)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
