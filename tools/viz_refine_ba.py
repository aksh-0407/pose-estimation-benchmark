#!/usr/bin/env python3
"""Before/after interactive 3D viewer for the 07 (refinement) stage.

Reads a player's ``pose_3d.keypoints_world_m`` from a BEFORE run dir (e.g. 06_roles) and
an AFTER run dir (e.g. 07_refine) and writes a self-contained, offline HTML per delivery
with two side-by-side animated 3D skeletons (before | after), a play button and a frame
slider. Colour per global_player_id. Nothing external is fetched (plotly.js is inlined).

    python3 tools/viz_refine_ba.py \
        --before-tree data/derived/8_init/pipetrack_v90_stack1f \
        --after-tree  data/derived/8_init/refine_demo \
        --out-dir     data/viz/refine_demo --step 2
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from core.keypoints import HALPE26_EDGES  # noqa: E402

PALETTE = ["#e6194B", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
           "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990"]


def _load(run_dir: Path, delivery: str) -> dict[str, dict[int, np.ndarray]]:
    """{gid: {frame_index: (26,3) array with NaN for missing joints}} for one delivery."""
    per_id: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for path in sorted((run_dir / delivery).rglob("predictions/*.jsonl")):
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            frame = int(rec["frame_index"])
            for pl in rec.get("players", []):
                gid = pl.get("global_player_id")
                pose = pl.get("pose_3d")
                if not gid or not pose or frame in per_id[gid]:
                    continue
                pts = np.array(
                    [[np.nan] * 3 if k is None else k for k in pose["keypoints_world_m"]],
                    dtype=float,
                )
                per_id[gid][frame] = pts
    return per_id


def _edge_xyz(pts: np.ndarray):
    xs, ys, zs = [], [], []
    for a, b in HALPE26_EDGES:
        if np.isfinite(pts[a]).all() and np.isfinite(pts[b]).all():
            xs += [pts[a, 0], pts[b, 0], None]
            ys += [pts[a, 1], pts[b, 1], None]
            zs += [pts[a, 2], pts[b, 2], None]
    return xs, ys, zs


def _bounds(*sources) -> tuple[np.ndarray, np.ndarray]:
    allpts = []
    for src in sources:
        for by_frame in src.values():
            for pts in by_frame.values():
                allpts.append(pts[np.isfinite(pts).all(axis=1)])
    stacked = np.vstack([a for a in allpts if len(a)]) if allpts else np.zeros((1, 3))
    lo, hi = stacked.min(axis=0), stacked.max(axis=0)
    span = float((hi - lo).max()) or 1.0
    mid = (hi + lo) / 2
    return mid - span / 2, mid + span / 2  # equal-aspect cube


def build_delivery_html(before, after, delivery: str, out_path: Path, step: int) -> None:
    gids = sorted(set(before) | set(after))
    colors = {g: PALETTE[i % len(PALETTE)] for i, g in enumerate(gids)}
    frames_all = sorted(
        set().union(*[set(before.get(g, {})) for g in gids],
                    *[set(after.get(g, {})) for g in gids])
    )
    frames_all = frames_all[::step]
    if not frames_all:
        return
    lo, hi = _bounds(before, after)

    fig = make_subplots(
        rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("BEFORE (06_roles)", "AFTER (07_refine - physics + smoothing)"),
        horizontal_spacing=0.02,
    )

    def traces_for(frame_idx: int):
        """One line + one marker trace per gid per panel; fixed order for animation."""
        data = []
        for src, scene in ((before, "scene"), (after, "scene2")):
            for g in gids:
                pts = src.get(g, {}).get(frame_idx)
                if pts is None:
                    pts = np.full((26, 3), np.nan)
                ex, ey, ez = _edge_xyz(pts)
                data.append(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                                         line=dict(color=colors[g], width=4),
                                         scene=scene, showlegend=False, hoverinfo="skip"))
                fin = np.isfinite(pts).all(axis=1)
                data.append(go.Scatter3d(x=pts[fin, 0], y=pts[fin, 1], z=pts[fin, 2],
                                         mode="markers", marker=dict(color=colors[g], size=3),
                                         scene=scene, showlegend=False, hoverinfo="skip"))
        return data

    for tr in traces_for(frames_all[0]):
        fig.add_trace(tr, row=1, col=1 if tr.scene == "scene" else 2)

    fig.frames = [go.Frame(name=str(f), data=traces_for(f)) for f in frames_all]

    scene_kw = dict(
        xaxis=dict(range=[lo[0], hi[0]]), yaxis=dict(range=[lo[1], hi[1]]),
        zaxis=dict(range=[lo[2], hi[2]]), aspectmode="cube",
    )
    fig.update_layout(
        title=f"{delivery} - 3D skeleton before vs after refinement ({len(gids)} players)",
        scene=scene_kw, scene2=scene_kw, template="plotly_dark", height=720,
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=0, xanchor="right",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=40, redraw=True),
                                                    fromcurrent=True, mode="immediate")]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], dict(frame=dict(duration=0, redraw=False),
                                                      mode="immediate")]),
                          ])],
        sliders=[dict(steps=[dict(method="animate", label=str(f),
                                  args=[[str(f)], dict(mode="immediate",
                                                       frame=dict(duration=0, redraw=True))])
                             for f in frames_all], y=0, x=0.1, len=0.85)],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_path), include_plotlyjs="inline", auto_play=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before-tree", required=True)
    ap.add_argument("--after-tree", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--deliveries", nargs="*", default=None)
    ap.add_argument("--step", type=int, default=2, help="frame subsample (keep HTML light)")
    args = ap.parse_args()

    before_tree, after_tree, out_dir = Path(args.before_tree), Path(args.after_tree), Path(args.out_dir)
    delivs = args.deliveries or sorted(
        d.name for d in after_tree.iterdir() if d.is_dir() and (d).rglob("predictions")
    )
    for delivery in delivs:
        before = _load(before_tree, delivery)
        after = _load(after_tree, delivery)
        if not after:
            print(f"[skip] {delivery}: no AFTER data")
            continue
        out_path = out_dir / f"{delivery}_before_after.html"
        build_delivery_html(before, after, delivery, out_path, args.step)
        print(f"[ok] {delivery} -> {out_path}")
    print(f"\nOpen the HTML files in a browser: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
