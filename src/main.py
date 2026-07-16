"""Full-pipeline batch driver: stabilization -> tracking -> association -> 3D lift ->
global_id -> roles -> mosaic render.

Extends the identity inner-loop driver (``identity.id_pipeline``) to the whole delivery
chain. P1 lands per delivery as the ``00_inference`` stage; each stage writes the canonical
run-dir layout ``<output-tree>/<DELIVERY>/{00_inference,01_stabilization,02_tracking,
03_association,04_lift,05_global_id,06_roles,logs}`` and mosaics land in ``<artifacts-root>/<DELIVERY>/``.

The 3D lift (04_lift) is the single triangulation and runs BEFORE global_id (Associate ->
Triangulate -> Track): global_id and roles carry its 3D forward, and 06_roles emits the
terminal role-stamped, suppression-filtered predictions consumed downstream.

Designed as the A/B workhorse (docs/critical-analysis/, wip/to_do.md):

- ``--from-stage``/``--until-stage`` (or ``--only``/``--skip`` if wired) select the stage
  window; ``--base-tree`` reuses upstream stage dirs from a frozen run (read in place).
- Every stage's config path and sha256 are recorded in ``pipeline_manifest.json``.
- ``--panel-only`` re-prints the joint metric panel; ``--baseline`` diffs it
  against a frozen snapshot tree (same layout, metrics files only).

Example (full chain on the reference delivery, run under the ``pose-lab`` env)::

    python -m main --dataset 8_init --version 9 \
        --deliveries CCPL080626M1_1_14_1 \
        --jobs 8 --p2-max-workers 2 --render-jobs 2

Example (association+ experiment reusing a frozen tree's tracking)::

    python -m main \
        --from-stage 03_association --base-tree data/derived/runs/pipetrack_v8 \
        --output-tree data/derived/runs/expt \
        --p3-config configs/03_association.yaml \
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from core.datasets import derived_root, raw_root, viz_root  # noqa: E402
from identity.id_pipeline import (  # noqa: E402
    ALL_DELIVERIES,
    _dig,
    _fmt,
    _run_stage,
)

STAGE_ORDER = ["01_stabilization", "02_tracking", "03_association", "04_lift", "05_global_id", "06_roles", "08_render"]

# Columns are read jointly — no single one is optimized in isolation.
# (name, metrics file relative to deliveries/<D>/, dotted key, format)
PANEL_COLUMNS = [
    ("agreement", "05_global_id/global_id_metrics.json", "cross_camera_agreement_rate", "{:.3f}"),
    ("ids", "05_global_id/global_id_metrics.json", "distinct_global_id_count", "{:d}"),
    ("teleports", "05_global_id/global_id_metrics.json", "teleport_event_count", "{:d}"),
    ("id_persist", "05_global_id/global_id_metrics.json",
     "completeness.confirmed_frame_completeness.mean", "{:.3f}"),
    ("frags", "05_global_id/global_id_metrics.json", "excess_id_fragment_count_proxy", "{:d}"),
    ("collisions", "05_global_id/global_id_metrics.json", "same_camera_identity_collision_frames", "{:d}"),
    ("coloc", "05_global_id/global_id_metrics.json", "colocated_disjoint_pair_count", "{:d}"),
    ("p2_tracks", "02_tracking/tracking_metrics.json", "@sum_confirmed_tracks", "{:d}"),
    ("single_cam", "03_association/association_metrics.json", "single_camera_rate", "{:.3f}"),
    ("churn", "03_association/association_metrics.json", "pair_link_churn_rate", "{:.3f}"),
    ("cycle_cons", "03_association/association_metrics.json", "cycle_consistency_rate", "{:.3f}"),
    ("chimera", "04_lift/triangulation_metrics.json", "chimera_suspect_count", "{:d}"),
    ("d_app", "03_association/association_metrics.json", "cue_d_prime.appearance", "{:.2f}"),
    ("jitter_px", "01_stabilization/stabilization_metrics.json", "mean_jitter_px_after", "{:.2f}"),
    ("tri_reproj", "04_lift/triangulation_metrics.json", "mean_reprojection_error_px", "{:.1f}"),
    ("tri_cov", "04_lift/triangulation_metrics.json", "triangulation_coverage", "{:.3f}"),
    ("verdict", "05_global_id/global_id_metrics.json", "quality_verdict.verdict", "{}"),
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
        self.output_root = Path(args.output_tree).resolve() / delivery
        self.base_root = (
            Path(args.base_tree).resolve() / delivery if args.base_tree else None
        )
        self.logs = self.output_root / "logs"

    def stage_dir(self, stage: str) -> Path:
        """The dir a stage writes to (output tree) or is reused from (base tree).

        In-window stages live in the output tree. Pre-window stages (notably ``00_inference``,
        which P1 writes into the run before this driver runs) are reused from ``--base-tree``
        when present, else read from the output tree itself.
        """
        if stage in self.stages:
            return self.output_root / stage
        if self.base_root is not None:
            reused = self.base_root / stage
            if reused.is_dir():
                return reused
        local = self.output_root / stage
        if local.is_dir():
            return local
        raise SystemExit(
            f"{self.delivery}: stage '{stage}' not found in the run"
            + ("" if self.base_root is None else " or --base-tree")
        )

    def p2_input(self) -> Path:
        """P2 reads stabilized predictions when 01 (stabilization) is enabled, else raw P1 (00_inference)."""
        if self.args.enable_stabilization:
            return self.stage_dir("01_stabilization")
        return self.stage_dir("00_inference")


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
        if stage == "08_render":
            continue
        out_dir = plan.output_root / stage
        log = plan.logs / f"{stage}.log"
        if stage == "01_stabilization":
            if not args.enable_stabilization:
                continue
            rc = _run_stage(
                "identity.p1_stabilization.run_stabilization",
                common(plan.stage_dir("00_inference"), out_dir)
                + ["--config", args.p1b_config],
                args.python, log,
            )
        elif stage == "02_tracking":
            rc = _run_stage(
                "identity.p2_tracking.run_per_camera_tracking",
                common(plan.p2_input(), out_dir)
                + ["--config", args.p2_config,
                   "--expected-frames", str(args.expected_frames),
                   "--max-workers", str(args.p2_max_workers)],
                args.python, log,
            )
        elif stage == "03_association":
            rc = _run_stage(
                "identity.p3_association.run_cross_camera_association",
                common(plan.stage_dir("02_tracking"), out_dir)
                + ["--config", args.p3_config,
                   "--expected-frames", str(args.expected_frames)],
                args.python, log,
            )
        elif stage == "04_lift":
            if not args.enable_lift:
                continue
            rc = _run_stage(
                "identity.p4_lift.run_triangulation",
                common(plan.stage_dir("03_association"), out_dir)
                + ["--id-source", "binding",
                   "--reprojection-threshold-px", str(args.tri_reproj_px),
                   "--min-views", str(args.tri_min_views),
                   "--ema-alpha", str(args.tri_ema_alpha),
                   "--smoother", args.tri_smoother,
                   "--butter-cutoff-hz", str(args.tri_butter_cutoff_hz)]
                + (["--cheirality"] if args.tri_cheirality else [])
                + (["--native-skeleton"] if args.tri_native_skeleton else [])
                + (["--dense-fill"] if args.tri_dense_fill else [])
                + (["--robust-refit", "--robust-huber-px", str(args.tri_robust_huber_px)]
                   if args.tri_robust_refit else []),
                args.python, log,
            )
        elif stage == "05_global_id":
            # Global-id reads the 04 lift run (single triangulation before identity, which
            # carries pose_3d forward); falls back to 03 only when the lift is disabled.
            gid_input = plan.stage_dir("04_lift") if args.enable_lift else plan.stage_dir("03_association")
            rc = _run_stage(
                "identity.p5_global_id.run_global_id",
                common(gid_input, out_dir)
                + ["--config", args.p4_config,
                   "--expected-frames", str(args.expected_frames)],
                args.python, log,
            )
        elif stage == "06_roles":
            rc = _run_stage(
                "identity.p6_roles.run_role_assignment",
                common(plan.stage_dir("05_global_id"), out_dir)
                + (["--config", args.p5_config] if args.p5_config else []),
                args.python, log,
            )
            if rc == 0:
                # Wave-6 (P5b): role-aware peripheral suppression. Explicit paths so a
                # reused base-tree p4 never makes the probe read the wrong p5 dir.
                rc = _run_stage(
                    "identity.p6_roles.suppress_peripherals",
                    ["--input-run-dir", str(plan.stage_dir("05_global_id")),
                     "--roles-path", str(out_dir / "roles.json"),
                     "--output-path", str(out_dir / "suppression.json")]
                    + (["--config", args.p5_config] if args.p5_config else []),
                    args.python, log,
                )
        else:  # pragma: no cover - registry and loop must stay in sync
            raise AssertionError(stage)
        result[f"{stage}_rc"] = rc
        # P3/P4 exit 1 for a warn/fail *verdict* but produced full output; every
        # other stage's nonzero rc means the stage itself failed -> stop the chain.
        if rc not in (0, 1) or (rc == 1 and stage not in ("03_association", "05_global_id")):
            result["failed_stage"] = stage
            return result
        # H7: a crashed P3/P4 ALSO exits 1 (uncaught exception) — distinguish a
        # warn-verdict from a crash by requiring the stage's metrics artifact.
        metrics_name = {"03_association": "association_metrics.json", "05_global_id": "global_id_metrics.json"}.get(stage)
        if rc == 1 and metrics_name and not (out_dir / metrics_name).exists():
            result["failed_stage"] = stage
            return result
    return result


def run_render(plan: DeliveryPlan) -> int:
    args, delivery = plan.args, plan.delivery
    artifact_dir = Path(args.artifacts_root).resolve() / delivery
    return _run_stage(
        "identity.visualization.render_videos",
        ["--run-dir", str(plan.stage_dir("05_global_id")), "--drive-root", args.drive_root,
         "--delivery-id", delivery, "--artifact-dir", str(artifact_dir),
         "--mode", "mosaic", "--show", "p4"],
        args.python, plan.logs / "render.log",
    )


def write_pipeline_manifest(args: argparse.Namespace, stages: list[str], deliveries: list[str]) -> None:
    configs = {
        "01_stabilization": args.p1b_config, "02_tracking": args.p2_config,
        "03_association": args.p3_config, "05_global_id": args.p4_config, "06_roles": args.p5_config or None,
    }
    manifest = {
        "schema_version": "pipeline_manifest/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_tree": str(Path(args.output_tree).resolve()),
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
            path = tree / delivery / rel
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
                        help="Comma-separated delivery ids; 'all' discovers every delivery "
                             "in the input tree's predictions/ (default: the 8 benchmark ids).")
    parser.add_argument("--data-root", default=None,
                        help="Base dir for raw/derived/viz (default: $PIPETRACK_DATA or 'data'; "
                             "on the L40S box point it at ~/bits-pose-data).")
    parser.add_argument("--dataset", default=None,
                        help="Dataset from configs/datasets.yaml (e.g. 8_init, 40_full). With "
                             "--version, derives --drive-root/--output-tree/--artifacts-root.")
    parser.add_argument("--version", default=None,
                        help="Run version token -> pipetrack_v<version> (requires --dataset).")
    parser.add_argument("--output-tree", default=None,
                        help="Run root; stages write <output-tree>/<DELIVERY>/<stage>/. P1 must have "
                             "written <output-tree>/<DELIVERY>/00_inference/. "
                             "Default with --dataset: <derived>/pipetrack_v<version>.")
    parser.add_argument("--base-tree", default=None,
                        help="Frozen tree to reuse stages before --from-stage from (read in place).")
    parser.add_argument("--from-stage", default="01_stabilization", choices=STAGE_ORDER)
    parser.add_argument("--until-stage", default="08_render", choices=STAGE_ORDER)
    parser.add_argument("--skip-render", action="store_true",
                        help="Shorthand for --until-stage 06_roles (skip the mosaic render).")
    parser.add_argument("--enable-stabilization", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Run 01 (stabilization) before P2 (v7 default ON; --no-enable-stabilization for v6-style runs).")
    parser.add_argument("--enable-lift", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Run the 04 (binding lift) binding-keyed 3D lift after P3 (v7 default ON).")
    parser.add_argument("--p1b-config", default="configs/01_stabilization.yaml")
    parser.add_argument("--p2-config", default="configs/02_tracking.yaml")
    parser.add_argument("--p3-config", default="configs/03_association.yaml")
    parser.add_argument("--p4-config", default="configs/05_global_id.yaml")
    parser.add_argument("--p5-config", default="configs/06_roles.yaml",
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
    parser.add_argument("--tri-robust-refit", action=argparse.BooleanOptionalAction, default=False,
                        help="Phase-1C: IRLS-Huber M-estimator polish on the per-joint triangulation "
                             "(down-weights marginal-inlier cameras). Off = byte-identical.")
    parser.add_argument("--tri-robust-huber-px", type=float, default=8.0,
                        help="Huber threshold (px) for --tri-robust-refit.")
    parser.add_argument("--tri-dense-fill", action=argparse.BooleanOptionalAction, default=True,
                        help="Fix C6: gap-gate temporal fills on real frame numbers (default off).")
    parser.add_argument("--artifacts-root", default=None,
                        help="Mosaics land in <artifacts-root>/<D>/ (required to render). "
                             "Default with --dataset: <DATA_ROOT>/viz/<dataset>/pipetrack_v<version>.")
    parser.add_argument("--drive-root", default=None,
                        help="Dataset raw/footage root. Default with --dataset: <DATA_ROOT>/raw/<dataset>.")
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
    # Dataset abstraction: one layout under DATA_ROOT, identical on every machine.
    # --dataset + --version derive the footage/derived/viz trees; explicit flags win.
    if args.dataset:
        if not args.version:
            raise SystemExit("--dataset requires --version (-> pipetrack_v<version>)")
        derived = str(derived_root(args.data_root, args.dataset, args.version))
        if args.drive_root is None:
            args.drive_root = str(raw_root(args.data_root, args.dataset))
        if args.output_tree is None:
            args.output_tree = derived
        if args.artifacts_root is None:
            args.artifacts_root = str(viz_root(args.data_root, args.dataset, args.version))
    if args.drive_root is None:
        args.drive_root = "drive"
    if args.output_tree is None:
        raise SystemExit("provide --output-tree (or --dataset with --version)")
    if args.skip_render and args.until_stage == "08_render":
        args.until_stage = "06_roles"
    if args.deliveries == "all":
        # Discover every delivery with P1 output in the run (<run>/<DELIVERY>/00_inference).
        run_root = Path(args.output_tree).resolve()
        seen = sorted(
            d.name for d in run_root.glob("*")
            if (d / "00_inference" / "predictions").is_dir()
        )
        if not seen:
            raise SystemExit(
                f"--deliveries all: no <DELIVERY>/00_inference/predictions found under {run_root}"
            )
        deliveries = seen
    else:
        deliveries = (
            [d.strip() for d in args.deliveries.split(",") if d.strip()]
            if args.deliveries else list(ALL_DELIVERIES)
        )
    stages = _stage_window(args.from_stage, args.until_stage)
    do_render = "08_render" in stages
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
