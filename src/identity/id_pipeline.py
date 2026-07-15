"""Batch driver for the PipeTrack identity portion (P3 -> P4) on all deliveries.

Re-runs cross-camera association (P3) and global-ID tracking (P4) for one or more
deliveries, reusing the on-disk P2 tracklets, then prints a joint metric panel and
(optionally) diffs it against a frozen baseline snapshot. BLAS threads are capped
per stage so eight deliveries can fan out across cores without oversubscription
(the lesson logged in ``wip/3d_location_methods_log.md``).

Example (all 8, reuse a tree's tracking -> association/global_id, diff a baseline)::

    python -m identity.id_pipeline \
        --input-tree data/derived/runs/pipetrack_v8 \
        --output-tree data/derived/runs/pipetrack_v8-id \
        --baseline data/derived/runs/pipetrack_v8/_baseline_snapshot \
        --jobs 8

``--panel-only`` skips running and just reads whatever metrics already exist under
the output tree (useful to re-print the table).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

ALL_DELIVERIES = [
    "CCPL080626M1_1_14_1",
    "CCPL080626M1_1_14_2",
    "CCPL080626M1_1_14_3",
    "CCPL080626M1_1_14_4",
    "CCPL080626M1_1_14_5",
    "CCPL080626M1_1_14_6",
    "CCPL080626M1_1_14_7",
    "CCPL080626M2_1_12_1",
]

# Columns read jointly — no single one is optimized in isolation.
PANEL_COLUMNS = [
    ("agreement", "cross_camera_agreement_rate", "{:.3f}"),
    ("ids", "distinct_global_id_count", "{:d}"),
    ("teleports", "teleport_event_count", "{:d}"),
    ("collisions", "same_camera_identity_collision_frames", "{:d}"),
    ("single_cam", "single_camera_rate", "{:.3f}"),          # from P3 metrics
    ("churn", "pair_link_churn_rate", "{:.3f}"),             # from P3 metrics
    ("verdict", "quality_verdict.verdict", "{}"),
]


def _blas_capped_env() -> dict[str, str]:
    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    env["PYTHONPATH"] = str(ROOT / "src")
    return env


def _run_stage(module: str, args: list[str], python: str, log: Path) -> int:
    cmd = [python, "-m", module, *args]
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            cmd, cwd=str(ROOT), env=_blas_capped_env(),
            stdout=handle, stderr=subprocess.STDOUT, text=True,
        )
    return proc.returncode


def run_delivery(
    delivery: str,
    *,
    input_tree: Path,
    output_tree: Path,
    drive_root: str,
    p3_config: str,
    p4_config: str,
    expected_frames: int,
    python: str,
    skip_p3: bool,
) -> dict:
    p2_dir = input_tree / "deliveries" / delivery / "02_tracking"
    p3_dir = output_tree / "deliveries" / delivery / "03_association"
    p4_dir = output_tree / "deliveries" / delivery / "05_global_id"
    logs = output_tree / "deliveries" / delivery / "logs"
    result = {"delivery": delivery, "p3_rc": None, "p4_rc": None}

    if not skip_p3:
        result["p3_rc"] = _run_stage(
            "identity.p3_association.run_cross_camera_association",
            ["--input-run-dir", str(p2_dir), "--output-run-dir", str(p3_dir),
             "--drive-root", drive_root, "--delivery-id", delivery,
             "--config", p3_config, "--expected-frames", str(expected_frames)],
            python, logs / "p3.log",
        )
        if result["p3_rc"] not in (0, 1):  # 1 = warn/fail verdict but ran
            return result
    result["p4_rc"] = _run_stage(
        "identity.p5_global_id.run_global_id",
        ["--input-run-dir", str(p3_dir), "--output-run-dir", str(p4_dir),
         "--drive-root", drive_root, "--delivery-id", delivery,
         "--config", p4_config, "--expected-frames", str(expected_frames)],
        python, logs / "p4.log",
    )
    return result


def _dig(payload: dict, dotted: str):
    node = payload
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def read_panel_row(tree: Path, delivery: str) -> dict:
    row: dict = {"delivery": delivery}
    p3 = tree / "deliveries" / delivery / "03_association" / "association_metrics.json"
    p4 = tree / "deliveries" / delivery / "05_global_id" / "global_id_metrics.json"
    p3_metrics = json.loads(p3.read_text()) if p3.exists() else {}
    p4_metrics = json.loads(p4.read_text()) if p4.exists() else {}
    for name, key, _fmt in PANEL_COLUMNS:
        source = p3_metrics if key in ("single_camera_rate", "pair_link_churn_rate") else p4_metrics
        row[name] = _dig(source, key)
    return row


def _fmt(value, spec: str) -> str:
    if value is None:
        return "-"
    try:
        if spec.endswith("d}"):
            return spec.format(int(value))
        if spec.endswith("f}"):
            return spec.format(float(value))
    except (TypeError, ValueError):
        return str(value)
    return spec.format(value)


def print_panel(rows: list[dict], baseline_rows: dict[str, dict] | None) -> None:
    headers = ["delivery"] + [name for name, _k, _f in PANEL_COLUMNS]
    print("\n| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for row in rows:
        cells = [row["delivery"]]
        for name, _key, spec in PANEL_COLUMNS:
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
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--deliveries", default=None,
                        help="Comma-separated delivery ids (default: all 8).")
    parser.add_argument("--input-tree", default="data/derived/runs/pipetrack_v8",
                        help="Tree holding the P2 inputs (deliveries/<D>/02_tracking).")
    parser.add_argument("--output-tree", default="data/derived/runs/pipetrack_v8-id",
                        help="Tree to write association/global_id into.")
    parser.add_argument("--drive-root", default="data/raw/8_init",
                        help="Dataset raw/footage root (default: data/raw/8_init).")
    parser.add_argument("--p3-config", default="configs/03_association.yaml")
    parser.add_argument("--p4-config", default="configs/05_global_id.yaml")
    parser.add_argument("--expected-frames", type=int, default=600)
    parser.add_argument("--python", default=sys.executable,
                        help="Interpreter for the pipeline stages (use the pose-lab env).")
    parser.add_argument("--jobs", type=int, default=4, help="Parallel deliveries.")
    parser.add_argument("--skip-p3", action="store_true",
                        help="Reuse existing P3 output; only re-run P4.")
    parser.add_argument("--panel-only", action="store_true",
                        help="Do not run; just read + print the metric panel.")
    parser.add_argument("--baseline", default=None,
                        help="Snapshot tree to diff the panel against.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    deliveries = (
        [d.strip() for d in args.deliveries.split(",") if d.strip()]
        if args.deliveries else list(ALL_DELIVERIES)
    )
    input_tree = (ROOT / args.input_tree).resolve()
    output_tree = (ROOT / args.output_tree).resolve()

    if not args.panel_only:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {
                pool.submit(
                    run_delivery, delivery,
                    input_tree=input_tree, output_tree=output_tree,
                    drive_root=args.drive_root, p3_config=args.p3_config,
                    p4_config=args.p4_config, expected_frames=args.expected_frames,
                    python=args.python, skip_p3=args.skip_p3,
                ): delivery
                for delivery in deliveries
            }
            for future in concurrent.futures.as_completed(futures):
                res = future.result()
                print(f"[done] {res['delivery']}  p3_rc={res['p3_rc']} p4_rc={res['p4_rc']}",
                      flush=True)

    rows = [read_panel_row(output_tree, delivery) for delivery in deliveries]
    baseline_rows = None
    if args.baseline:
        baseline_tree = (ROOT / args.baseline).resolve()
        baseline_rows = {d: read_panel_row(baseline_tree, d) for d in deliveries}
    print_panel(rows, baseline_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
