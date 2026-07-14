"""Full-pipeline batch driver: P1.5 -> P2 -> P3 -> P4 -> P5 -> 3D lift -> mosaic render.

Extends the P3->P4 inner-loop driver (``run_id_pipeline``) to the whole delivery
chain starting from a P1 predictions run (e.g. ``benchmarks/runs/rtmpose-x``).
Each stage writes the canonical run-dir layout under
``<output-tree>/deliveries/<DELIVERY>/{p1b,p2,p3,p4,p5,p6_3d,logs}`` and the
mosaics land in ``<artifacts-root>/mosaics/<DELIVERY>/``.

Designed as the A/B workhorse for the fix campaign (docs/critical-analysis/to-do.md):

- ``--base-tree`` + ``--from-stage`` reuse upstream stage dirs from a frozen run
  so a P3-only experiment never re-runs P2 (no copies, dirs are read in place).
- Every stage's config path and sha256 are recorded in ``pipeline_manifest.json``.
- ``--panel-only`` re-prints the joint metric panel; ``--baseline`` diffs it
  against a frozen snapshot tree (same layout, metrics files only).

Example (v6.0 ground baseline)::

    python -m scripts.pipetrack.run_full_pipeline \
        --input-tree benchmarks/runs/rtmpose-x \
        --output-tree benchmarks/runs/pipetrack_v6.0 \
        --artifacts-root artifacts/pipetrack_v6.0 \
        --p2-config configs/v6/p2_tracking.yaml \
        --p3-config configs/v6/p3_association.yaml \
        --p4-config configs/v6/p4_global_id.yaml \
        --jobs 8 --p2-max-workers 2 --render-jobs 2

Example (P3+ experiment reusing the frozen baseline's P2)::

    python -m scripts.pipetrack.run_full_pipeline \
        --from-stage p3 --base-tree benchmarks/runs/pipetrack_v6.0 \
        --output-tree benchmarks/runs/pipetrack_v6.1-f02 \
        --p3-config configs/experiments/v6_f02_c07__p3.yaml \
        --p4-config configs/v6/p4_global_id.yaml \
        --baseline benchmarks/runs/pipetrack_v6.0/_baseline_snapshot \
        --skip-render --jobs 8
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pipetrack.run_id_pipeline import (  # noqa: E402
    ALL_DELIVERIES,
    _dig,
    _fmt,
    _run_stage,
)

STAGE_ORDER = ["p1b", "p2", "p3", "p3_5", "p4", "p5", "p6_3d", "render"]

# Columns are read jointly — no single one is optimized in isolation.
# (name, metrics file relative to deliveries/<D>/, dotted key, format)
PANEL_COLUMNS = [
    ("agreement", "p4/global_id_metrics.json", "cross_camera_agreement_rate", "{:.3f}"),
    ("ids", "p4/global_id_metrics.json", "distinct_global_id_count", "{:d}"),
    ("teleports", "p4/global_id_metrics.json", "teleport_event_count", "{:d}"),
    ("id_persist", "p4/global_id_metrics.json",
     "completeness.confirmed_frame_completeness.mean", "{:.3f}"),
    ("frags", "p4/global_id_metrics.json", "excess_id_fragment_count_proxy", "{:d}"),
    ("collisions", "p4/global_id_metrics.json", "same_camera_identity_collision_frames", "{:d}"),
    ("p2_tracks", "p2/tracking_metrics.json", "@sum_confirmed_tracks", "{:d}"),
    ("single_cam", "p3/association_metrics.json", "single_camera_rate", "{:.3f}"),
    ("churn", "p3/association_metrics.json", "pair_link_churn_rate", "{:.3f}"),
    ("cycle_cons", "p3/association_metrics.json", "cycle_consistency_rate", "{:.3f}"),
    ("chimera", "p3_5/triangulation_metrics.json", "chimera_suspect_count", "{:d}"),
    ("d_app", "p3/association_metrics.json", "cue_d_prime.appearance", "{:.2f}"),
    ("jitter_px", "p1b/stabilization_metrics.json", "mean_jitter_px_after", "{:.2f}"),
    ("tri_reproj", "p6_3d/triangulation_metrics.json", "mean_reprojection_error_px", "{:.1f}"),
    ("tri_cov", "p6_3d/triangulation_metrics.json", "triangulation_coverage", "{:.3f}"),
    ("verdict", "p4/global_id_metrics.json", "quality_verdict.verdict", "{}"),
]


def _sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _stage_window(from_stage: str, until_stage: str) -> list[str]:
    start, stop = STAGE_ORDER.index(from_stage), STAGE_ORDER.index(until_stage)
    if start > stop:
        raise SystemExit(f"--from-stage {from_stage} is after --until-stage {until_stage}")
    return STAGE_ORDER[start : stop + 1]


class DeliveryPlan:
    """Resolves each stage's input/output dirs for one delivery."""

    def __init__(self, delivery: str, args: argparse.Namespace, stages: list[str]):
        self.delivery = delivery
        self.args = args
        self.stages = stages
        self.output_root = Path(args.output_tree).resolve() / "deliveries" / delivery
        self.base_root = (
            Path(args.base_tree).resolve() / "deliveries" / delivery if args.base_tree else None
        )
        self.logs = self.output_root / "logs"

    def stage_dir(self, stage: str) -> Path:
        """The dir a stage writes to (output tree) or is reused from (base tree)."""
        if stage in self.stages:
            return self.output_root / stage
        if self.base_root is None:
            raise SystemExit(
                f"{self.delivery}: stage '{stage}' is outside the run window and no "
                f"--base-tree was given to reuse it from"
            )
        reused = self.base_root / stage
        if not reused.is_dir():
            raise SystemExit(f"{self.delivery}: reused stage dir missing: {reused}")
        return reused

    def p2_input(self) -> Path:
        """P2 reads stabilized predictions when P1.5 is enabled, else the raw P1 run."""
        if self.args.enable_stabilization:
            return self.stage_dir("p1b")
        return Path(self.args.input_tree).resolve()


def run_compute_chain(plan: DeliveryPlan) -> dict:
    """Run every non-render stage in the window for one delivery. Returns rc per stage."""
    args, delivery = plan.args, plan.delivery
    result: dict = {"delivery": delivery}

    def common(input_dir: Path, output_dir: Path) -> list[str]:
        return [
            "--input-run-dir", str(input_dir), "--output-run-dir", str(output_dir),
            "--drive-root", args.drive_root, "--delivery-id", delivery,
        ]

    for stage in plan.stages:
        if stage == "render":
            continue
        out_dir = plan.output_root / stage
        log = plan.logs / f"{stage}.log"
        if stage == "p1b":
            if not args.enable_stabilization:
                continue
            rc = _run_stage(
                "scripts.stabilization.run_stabilization",
                common(Path(args.input_tree).resolve(), out_dir)
                + ["--config", args.p1b_config],
                args.python, log,
            )
        elif stage == "p2":
            rc = _run_stage(
                "scripts.tracking.run_per_camera_tracking",
                common(plan.p2_input(), out_dir)
                + ["--config", args.p2_config,
                   "--expected-frames", str(args.expected_frames),
                   "--max-workers", str(args.p2_max_workers)],
                args.python, log,
            )
        elif stage == "p3":
            rc = _run_stage(
                "scripts.association.run_cross_camera_association",
                common(plan.stage_dir("p2"), out_dir)
                + ["--config", args.p3_config,
                   "--expected-frames", str(args.expected_frames)],
                args.python, log,
            )
        elif stage == "p3_5":
            if not args.enable_lift:
                continue
            rc = _run_stage(
                "scripts.export.triangulate_predictions",
                common(plan.stage_dir("p3"), out_dir)
                + ["--id-source", "binding",
                   "--reprojection-threshold-px", str(args.tri_reproj_px),
                   "--min-views", str(args.tri_min_views),
                   "--ema-alpha", str(args.tri_ema_alpha),
                   "--smoother", args.tri_smoother,
                   "--butter-cutoff-hz", str(args.tri_butter_cutoff_hz)]
                + (["--cheirality"] if args.tri_cheirality else [])
                + (["--native-skeleton"] if args.tri_native_skeleton else [])
                + (["--dense-fill"] if args.tri_dense_fill else []),
                args.python, log,
            )
        elif stage == "p4":
            rc = _run_stage(
                "scripts.global_id.run_global_id",
                common(plan.stage_dir("p3"), out_dir)
                + ["--config", args.p4_config,
                   "--expected-frames", str(args.expected_frames)],
                args.python, log,
            )
        elif stage == "p5":
            rc = _run_stage(
                "scripts.roles.run_role_assignment",
                common(plan.stage_dir("p4"), out_dir)
                + (["--config", args.p5_config] if args.p5_config else []),
                args.python, log,
            )
            if rc == 0:
                # Wave-6 (P5b): role-aware peripheral suppression. Explicit paths so a
                # reused base-tree p4 never makes the probe read the wrong p5 dir.
                rc = _run_stage(
                    "scripts.roles.suppress_peripherals",
                    ["--input-run-dir", str(plan.stage_dir("p4")),
                     "--roles-path", str(out_dir / "roles.json"),
                     "--output-path", str(out_dir / "suppression.json")]
                    + (["--config", args.p5_config] if args.p5_config else []),
                    args.python, log,
                )
        elif stage == "p6_3d":
            rc = _run_stage(
                "scripts.export.triangulate_predictions",
                common(plan.stage_dir("p4"), out_dir)
                + ["--reprojection-threshold-px", str(args.tri_reproj_px),
                   "--min-views", str(args.tri_min_views),
                   "--ema-alpha", str(args.tri_ema_alpha),
                   "--smoother", args.tri_smoother,
                   "--butter-cutoff-hz", str(args.tri_butter_cutoff_hz)]
                + (["--cheirality"] if args.tri_cheirality else [])
                + (["--native-skeleton"] if args.tri_native_skeleton else [])
                + (["--dense-fill"] if args.tri_dense_fill else []),
                args.python, log,
            )
        else:  # pragma: no cover - registry and loop must stay in sync
            raise AssertionError(stage)
        result[f"{stage}_rc"] = rc
        # P3/P4 exit 1 for a warn/fail *verdict* but produced full output; every
        # other stage's nonzero rc means the stage itself failed -> stop the chain.
        if rc not in (0, 1) or (rc == 1 and stage not in ("p3", "p4")):
            result["failed_stage"] = stage
            return result
        # H7: a crashed P3/P4 ALSO exits 1 (uncaught exception) — distinguish a
        # warn-verdict from a crash by requiring the stage's metrics artifact.
        metrics_name = {"p3": "association_metrics.json", "p4": "global_id_metrics.json"}.get(stage)
        if rc == 1 and metrics_name and not (out_dir / metrics_name).exists():
            result["failed_stage"] = stage
            return result
    return result


def run_render(plan: DeliveryPlan) -> int:
    args, delivery = plan.args, plan.delivery
    artifact_dir = Path(args.artifacts_root).resolve() / "mosaics" / delivery
    return _run_stage(
        "scripts.visualization.render_phase1_videos",
        ["--run-dir", str(plan.stage_dir("p4")), "--drive-root", args.drive_root,
         "--delivery-id", delivery, "--artifact-dir", str(artifact_dir),
         "--mode", "mosaic", "--show", "p4"],
        args.python, plan.logs / "render.log",
    )


def write_pipeline_manifest(args: argparse.Namespace, stages: list[str], deliveries: list[str]) -> None:
    configs = {
        "p1b": args.p1b_config, "p2": args.p2_config,
        "p3": args.p3_config, "p4": args.p4_config, "p5": args.p5_config or None,
    }
    manifest = {
        "schema_version": "pipeline_manifest/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_tree": str(Path(args.input_tree).resolve()),
        "base_tree": str(Path(args.base_tree).resolve()) if args.base_tree else None,
        "stages_run": stages,
        "deliveries": deliveries,
        "enable_stabilization": args.enable_stabilization,
        "enable_lift": args.enable_lift,
        "configs": {
            stage: ({"path": path, "sha256": _sha256(ROOT / path)} if path else None)
            for stage, path in configs.items()
        },
        "triangulation": {
            "reprojection_threshold_px": args.tri_reproj_px,
            "min_views": args.tri_min_views,
            "ema_alpha": args.tri_ema_alpha,
            "cheirality": args.tri_cheirality,
            "smoother": args.tri_smoother,
            "butter_cutoff_hz": args.tri_butter_cutoff_hz,
            "native_skeleton": args.tri_native_skeleton,
            "dense_fill": args.tri_dense_fill,
        },
    }
    out = Path(args.output_tree).resolve() / "pipeline_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sum_confirmed_tracks(metrics: dict):
    """Total confirmed per-camera tracks — the P2 fragmentation proxy (~13-15 people
    per camera view is ideal; excess = per-camera track fragments P4 must stitch)."""
    per_camera = metrics.get("per_camera")
    if not isinstance(per_camera, dict):
        return None
    values = [
        _dig(camera, "summary.confirmed_tracks") for camera in per_camera.values()
    ]
    values = [value for value in values if isinstance(value, (int, float))]
    return int(sum(values)) if values else None


_COMPUTED_COLUMNS = {"@sum_confirmed_tracks": _sum_confirmed_tracks}


def read_panel_row(tree: Path, delivery: str) -> dict:
    row: dict = {"delivery": delivery}
    cache: dict[str, dict] = {}
    for name, rel, key, _spec in PANEL_COLUMNS:
        if rel not in cache:
            path = tree / "deliveries" / delivery / rel
            cache[rel] = json.loads(path.read_text()) if path.exists() else {}
        if key.startswith("@"):
            row[name] = _COMPUTED_COLUMNS[key](cache[rel])
        else:
            row[name] = _dig(cache[rel], key)
    return row


def print_panel(rows: list[dict], baseline_rows: dict[str, dict] | None) -> None:
    headers = ["delivery"] + [name for name, _r, _k, _f in PANEL_COLUMNS]
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [row["delivery"]]
        for name, _rel, _key, spec in PANEL_COLUMNS:
            text = _fmt(row.get(name), spec)
            if baseline_rows and name != "verdict":
                base = baseline_rows.get(row["delivery"], {}).get(name)
                cur = row.get(name)
                if isinstance(base, (int, float)) and isinstance(cur, (int, float)):
                    delta = cur - base
                    if abs(delta) > (0.0005 if spec.endswith("f}") else 0):
                        arrow = "+" if delta > 0 else ""
                        text += f" ({arrow}{_fmt(delta, spec)})"
            cells.append(text)
        print("| " + " | ".join(cells) + " |")
    print()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--deliveries", default=None,
                        help="Comma-separated delivery ids (default: all 8).")
    parser.add_argument("--input-tree", default="benchmarks/runs/rtmpose-x-tiled-w5-full",
                        help="P1 predictions run dir (flat predictions/*.jsonl).")
    parser.add_argument("--output-tree", required=True,
                        help="Tree to write stage outputs into (deliveries/<D>/...).")
    parser.add_argument("--base-tree", default=None,
                        help="Frozen tree to reuse stages before --from-stage from (read in place).")
    parser.add_argument("--from-stage", default="p1b", choices=STAGE_ORDER)
    parser.add_argument("--until-stage", default="render", choices=STAGE_ORDER)
    parser.add_argument("--skip-render", action="store_true",
                        help="Shorthand for --until-stage p6_3d.")
    parser.add_argument("--enable-stabilization", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Run P1.5 before P2 (v7 default ON; --no-enable-stabilization for v6-style runs).")
    parser.add_argument("--enable-lift", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Run the P3.5 binding-keyed 3D lift after P3 (v7 default ON).")
    parser.add_argument("--p1b-config", default="configs/v8/p1b_stabilization.yaml")
    parser.add_argument("--p2-config", default="configs/v8/p2_tracking.yaml")
    parser.add_argument("--p3-config", default="configs/v8/p3_association.yaml")
    parser.add_argument("--p4-config", default="configs/v8/p4_global_id.yaml")
    parser.add_argument("--p5-config", default="configs/v8/p5_roles.yaml",
                        help="P5 roles YAML (v1.1 epoch solver); pass '' for legacy v0.")
    parser.add_argument("--tri-reproj-px", type=float, default=10.0)
    parser.add_argument("--tri-min-views", type=int, default=2)
    parser.add_argument("--tri-ema-alpha", type=float, default=0.65)
    parser.add_argument("--tri-cheirality", action=argparse.BooleanOptionalAction, default=True,
                        help="Fix F3: cheirality gate in the 3D lift (default off = baseline).")
    parser.add_argument("--tri-smoother", choices=["ema", "butterworth"], default="butterworth",
                        help="Fix F7: zero-phase Butterworth instead of causal EMA (default ema).")
    parser.add_argument("--tri-butter-cutoff-hz", type=float, default=6.0)
    parser.add_argument("--tri-native-skeleton", action=argparse.BooleanOptionalAction, default=True,
                        help="Fix F15: triangulate all 26 Halpe keypoints (default off = COCO-17).")
    parser.add_argument("--tri-dense-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="Fix C6: gap-gate temporal fills on real frame numbers (default off).")
    parser.add_argument("--artifacts-root", default=None,
                        help="Mosaics land in <artifacts-root>/mosaics/<D>/ (required to render).")
    parser.add_argument("--drive-root", default="drive")
    parser.add_argument("--expected-frames", type=int, default=600)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", type=int, default=4, help="Parallel delivery compute chains.")
    parser.add_argument("--p2-max-workers", type=int, default=2,
                        help="Per-delivery camera workers inside P2.")
    parser.add_argument("--render-jobs", type=int, default=2,
                        help="Parallel renders (each decodes 7 JPEG streams; keep small).")
    parser.add_argument("--panel-only", action="store_true",
                        help="Do not run; just read + print the metric panel.")
    parser.add_argument("--baseline", default=None,
                        help="Snapshot tree to diff the panel against.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.skip_render and args.until_stage == "render":
        args.until_stage = "p6_3d"
    deliveries = (
        [d.strip() for d in args.deliveries.split(",") if d.strip()]
        if args.deliveries else list(ALL_DELIVERIES)
    )
    stages = _stage_window(args.from_stage, args.until_stage)
    do_render = "render" in stages
    if do_render and not args.artifacts_root and not args.panel_only:
        raise SystemExit("--artifacts-root is required when the render stage is in the window")

    output_tree = (ROOT / args.output_tree).resolve()

    if not args.panel_only:
        write_pipeline_manifest(args, stages, deliveries)
        plans = {d: DeliveryPlan(d, args, stages) for d in deliveries}
        failures: list[str] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as compute, \
                concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.render_jobs)) as render:
            render_futures = {}

            def chain_then_render(delivery: str) -> dict:
                res = run_compute_chain(plans[delivery])
                if do_render and "failed_stage" not in res:
                    render_futures[delivery] = render.submit(run_render, plans[delivery])
                return res

            compute_futures = {
                compute.submit(chain_then_render, d): d for d in deliveries
            }
            for future in concurrent.futures.as_completed(compute_futures):
                res = future.result()
                rcs = " ".join(
                    f"{k[:-3]}={v}" for k, v in res.items() if k.endswith("_rc")
                )
                status = f"FAILED at {res['failed_stage']}" if "failed_stage" in res else "ok"
                if "failed_stage" in res:
                    failures.append(res["delivery"])
                print(f"[compute {status}] {res['delivery']}  {rcs}", flush=True)
            for delivery, future in render_futures.items():
                rc = future.result()
                print(f"[render {'ok' if rc == 0 else f'rc={rc}'}] {delivery}", flush=True)
        if failures:
            print(f"\n{len(failures)} deliveries failed: {', '.join(sorted(failures))}")

    rows = [read_panel_row(output_tree, d) for d in deliveries]
    baseline_rows = None
    if args.baseline:
        baseline_tree = (ROOT / args.baseline).resolve()
        baseline_rows = {d: read_panel_row(baseline_tree, d) for d in deliveries}
    print_panel(rows, baseline_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
