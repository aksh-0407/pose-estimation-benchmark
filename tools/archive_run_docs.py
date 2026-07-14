#!/usr/bin/env python3
"""Archive benchmark run trees as markdown documents before deletion.

For every run under data/derived/runs/ this emits docs/runs/<run>.md containing the
run's purpose/verdict (curated table below), its pipeline manifest (configs + shas +
base-tree lineage), and a per-delivery metric panel harvested from the stage metrics
JSONs. P1 inference runs get their p1_metrics.json summary instead. The goal: the
full analytical record survives even though the multi-GB predictions/logs are removed.

Usage:
    python tools/archive_run_docs.py [--runs-root data/derived/runs] \
        [--out docs/runs] [--only run1 run2 ...]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Curated context: purpose + verdict + fixes-log pointer per run name.
RUN_NOTES = {
    "pipetrack_v3": ("Historical identity stack (pre-campaign, v3 era).", "Superseded", "wip/methods_log.md"),
    "pipetrack_v5": ("Validated v5 identity stack on RTMPose-L data.", "Superseded", "wip/methods_log.md ID-0..6"),
    "pipetrack_v6.0": ("Frozen ground baseline of the fix campaign (v5 configs, RTMPose-X P1).", "Baseline (superseded by v8.0)", "fixes-log F0"),
    "pipetrack_v6.0.zip": ("Zip archive of pipetrack_v6.0.", "Redundant archive", "fixes-log F0"),
    "pipetrack_v6.1-f01": ("Wave-0 A/B: 01 stabilization wired (F1).", "Accepted into v7 lineage", "fixes-log F1"),
    "pipetrack_v6.1-wave1": ("Wave-1 correctness batch (F3-F8).", "Accepted into v7 lineage", "fixes-log W1"),
    "pipetrack_v6.2-wave3": ("Wave-3 stack (F9a covariance, F10 R, F11 shape, F12 posture stitch).", "Accepted into v7 lineage", "fixes-log W3"),
    "pipetrack_v6.2-wave3b": ("Wave-3b asymmetric-R refinement.", "Accepted into v7 lineage", "fixes-log W3b/W4"),
    "pipetrack_v6.3-wave4": ("Wave-4 chimera splitting (F13).", "Accepted into v7 lineage", "fixes-log W3b/W4"),
    "pipetrack_v7-rc1": ("First composed v7 release candidate.", "Rejected (H3 binding collapse; root-caused)", "fixes-log v7-rc1"),
    "pipetrack_v7-rc2": ("Re-composed v7 on fixed code; stitcher live first time.", "Accepted as v7 default (superseded by v8.0)", "fixes-log v7-rc2 + GRAND ANALYSIS"),
    "pipetrack_v7-rc3": ("01 (stabilization) isolation (no stabilization).", "Rejected (worse worst-clip floor)", "fixes-log GRAND ANALYSIS"),
    "pipetrack_v7-ablA": ("Wave ablation helper tree.", "Diagnostic only", "fixes-log W3b/W4"),
    "pipetrack_v7-w5b": ("Contested-camera weighting composed A/B.", "No-op proven (P1 NMS 0.3 caps same-cam IoU)", "fixes-log W5B"),
    "pipetrack_v8-probe": ("Phase C: tiled NMS-0.3 P1 through v7 stack (_7+M2).", "Hold verdict; superseded by nms55", "fixes-log W5-C"),
    "pipetrack_v8-nms55w5b": ("Tiled NMS-0.55 + contested-0.30 (_7+M2).", "Contested rejected (-0.08 agreement)", "fixes-log W5B-LIVE"),
    "pipetrack_v8-nms55only": ("Tiled NMS-0.55 ablation without contested (_7+M2).", "Winner; became v8 detection spec", "fixes-log W5B-LIVE"),
    "_v8_nospawn_probe": ("lowconf_can_spawn=false probe (_5,_6,_7,M2).", "Adopted into v8.0 (strict improvement)", "fixes-log GRAND ANALYSIS v2"),
    "_w5b_id_check": ("W5b flags-off byte-identity check tree (M2).", "Diagnostic only (identity proven)", "fixes-log W5B"),
    "pipetrack_v8-rc1": ("Composed v8 candidate: tiled+NMS55 x8, v7 stack, W6.", "Superseded by v8.0 (adds no-spawn)", "fixes-log GRAND ANALYSIS v2"),
    "pipetrack_v8.0": ("ACCEPTED v8.0 default tree (KEPT).", "Current default", "fixes-log GRAND ANALYSIS v2"),
    "rtmpose-x": ("RTMPose-X P1 at plain 640 detection (8 deliveries).", "Superseded by tiled-w5-full", "fixes-log F0/W5"),
    "rtmpose-x.zip": ("Zip archive of rtmpose-x.", "Redundant archive", "-"),
    "rtmpose-l-body8-full-db32-pb96": ("RTMPose-L body8 P1 full run.", "Rejected (X chosen for Halpe-26 accuracy)", "wip/model_comparison.md"),
    "rtmpose-x-tiled-w5": ("Tiled NMS-0.3 P1 probe (_7+M2).", "Superseded by nms55", "fixes-log W5-C"),
    "rtmpose-x-tiled-nms55": ("Tiled NMS-0.55 P1 probe (_7+M2).", "Superseded by tiled-w5-full", "fixes-log W5B-LIVE"),
    "rtmpose-x-tiled-w5-full": ("Tiled NMS-0.55 P1, all 8 benchmark deliveries (KEPT - v8 input).", "Current best RTMPose P1", "fixes-log GRAND ANALYSIS v2"),
    "yolo26x-pose-full-db8": ("YOLO26x-pose P1 full run (KEPT - best YOLO data).", "Kept for model comparison", "wip/model_comparison.md"),
    "pipetrack_v8.1-w9": ("ACCEPTED v8.1 reference (KEPT): W9 union-lift + colocated merges over v8.0.", "Current local reference", "fixes-log W9"),
    "_w9_probe": ("W9 union-lift iteration probe (_7,_2,_6).", "Diagnostic; superseded by v8.1-w9", "fixes-log W9"),
    "_w9_probe2": ("W9 gate-widening probe (_7).", "Diagnostic", "fixes-log W9"),
    "_w9_probe3": ("W9 rejection-counter probe (_7).", "Diagnostic", "fixes-log W9"),
    "_w9_id_check": ("W9 flags-off byte-identity tree (M2).", "Diagnostic (identity proven)", "fixes-log W9"),
    "_p3_prefetch_check": ("P3 appearance-prefetch byte-identity tree (M2).", "Diagnostic (identity proven)", "fixes-log W10-PERF"),
    "bakeoff_w5": ("Detector-only recall bake-off (5 candidates, _7+M2 sampled).", "t640 tiled won; native hi-res dead", "fixes-log W5"),
}

PANEL_FIELDS = [
    ("agreement", ("05_global_id", "global_id_metrics.json"), "cross_camera_agreement_rate"),
    ("ids", ("05_global_id", "global_id_metrics.json"), "distinct_global_id_count"),
    ("teleports", ("05_global_id", "global_id_metrics.json"), "teleport_event_count"),
    ("id_persist", ("05_global_id", "global_id_metrics.json"), "completeness.confirmed_frame_completeness.mean"),
    ("frags", ("05_global_id", "global_id_metrics.json"), "excess_id_fragment_count_proxy"),
    ("stitch_links", ("05_global_id", "global_id_metrics.json"), "selected_stitch_link_count"),
    ("tri_reproj_px", ("07_lift3d", "triangulation_metrics.json"), "mean_reprojection_error_px"),
    ("tri_cov", ("07_lift3d", "triangulation_metrics.json"), "triangulation_coverage"),
]


def _dig(payload: dict, dotted: str):
    value = payload
    for key in dotted.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def panel_rows(tree: Path) -> list[str]:
    rows = []
    for ddir in sorted((tree / "deliveries").glob("*")) if (tree / "deliveries").is_dir() else []:
        cache: dict[tuple, dict] = {}
        cells = []
        for _, (stage, fname), dotted in PANEL_FIELDS:
            key = (stage, fname)
            if key not in cache:
                path = ddir / stage / fname
                try:
                    cache[key] = json.loads(path.read_text()) if path.is_file() else {}
                except (OSError, json.JSONDecodeError):
                    cache[key] = {}
            cells.append(_fmt(_dig(cache[key], dotted)))
        if any(c != "-" for c in cells):
            rows.append(f"| {ddir.name} | " + " | ".join(cells) + " |")
    return rows


def manifest_section(tree: Path) -> list[str]:
    out = []
    mpath = tree / "pipeline_manifest.json"
    if mpath.is_file():
        try:
            m = json.loads(mpath.read_text())
        except (OSError, json.JSONDecodeError):
            return ["(pipeline_manifest.json unreadable)"]
        out.append(f"- created_at: {m.get('created_at')}")
        out.append(f"- base_tree: {m.get('base_tree')}")
        out.append(f"- stages_run: {m.get('stages_run')}")
        out.append(f"- stabilization/lift: {m.get('enable_stabilization')}/{m.get('enable_lift')}")
        for stage, cfg in (m.get("configs") or {}).items():
            if cfg:
                out.append(f"- config[{stage}]: `{cfg.get('path')}` sha256 `{str(cfg.get('sha256'))[:16]}…`")
        if m.get("triangulation"):
            out.append(f"- triangulation: {m['triangulation']}")
    p1m = tree / "p1_metrics.json"
    if p1m.is_file():
        try:
            m = json.loads(p1m.read_text())
            s = m.get("summary", {})
            out.append(f"- P1 model: {m.get('model_id')} | det/pose batch: "
                       f"{m.get('det_batch_size')}/{m.get('pose_batch_size')} | tiled: {m.get('tiled_det')}")
            out.append(f"- P1 summary: {json.dumps({k: s.get(k) for k in ('records_written', 'total_players_detected', 'wall_clock_s', 'fps_overall', 'status')})}")
        except (OSError, json.JSONDecodeError):
            out.append("(p1_metrics.json unreadable)")
    if not out:
        out.append("(no manifest found — pre-driver-era tree; directory listing below)")
        entries = sorted(p.name for p in tree.iterdir())[:20]
        out.append("- top-level entries: " + ", ".join(entries))
    return out


def archive_run(tree: Path, out_dir: Path) -> bool:
    name = tree.name
    purpose, verdict, pointer = RUN_NOTES.get(
        name, ("(uncurated run)", "unknown", "-"))
    lines = [f"# Run archive: `{name}`", ""]
    lines.append(f"- **Purpose:** {purpose}")
    lines.append(f"- **Verdict:** {verdict}")
    lines.append(f"- **Full analysis:** {pointer}")
    lines.append(f"- Archived: {datetime.now(timezone.utc).isoformat()} (data tree deleted after archival)")
    lines.append("")
    lines.append("## Manifest / provenance")
    lines += manifest_section(tree)
    rows = panel_rows(tree)
    if rows:
        lines.append("")
        lines.append("## Per-delivery metric panel")
        header = "| delivery | " + " | ".join(f[0] for f in PANEL_FIELDS) + " |"
        lines.append(header)
        lines.append("|" + "---|" * (len(PANEL_FIELDS) + 1))
        lines += rows
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}.md").write_text("\n".join(lines) + "\n")
    return bool(rows) or (tree / "p1_metrics.json").is_file() or not (tree / "deliveries").is_dir()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", default="data/derived/runs")
    ap.add_argument("--out", default="docs/runs")
    ap.add_argument("--only", nargs="+", default=None)
    args = ap.parse_args()
    runs_root = ROOT / args.runs_root
    out_dir = ROOT / args.out
    names = args.only or sorted(p.name for p in runs_root.iterdir())
    index = ["# Archived benchmark runs", "",
             "Historical run trees were documented here and their bulk data deleted "
             "(2026-07-14 cleanup). Full analytical narrative: docs/critical-analysis/fixes-log.md.",
             "", "| run | purpose | verdict | analysis pointer |", "|---|---|---|---|"]
    ok = True
    for name in names:
        tree = runs_root / name
        if name.endswith(".zip"):
            purpose, verdict, pointer = RUN_NOTES.get(name, ("zip archive", "redundant", "-"))
            index.append(f"| {name} | {purpose} | {verdict} | {pointer} |")
            continue
        if not tree.is_dir():
            continue
        good = archive_run(tree, out_dir)
        purpose, verdict, pointer = RUN_NOTES.get(name, ("(uncurated)", "unknown", "-"))
        index.append(f"| [{name}]({name}.md) | {purpose} | {verdict} | {pointer} |")
        status = "ok" if good else "EMPTY-PANEL"
        print(f"archived {name}: {status}")
        ok = ok and good
    (out_dir / "README.md").write_text("\n".join(index) + "\n")
    print("index written:", out_dir / "README.md")
    if not ok:
        raise SystemExit("some runs produced empty docs — inspect before deleting")


if __name__ == "__main__":
    main()
