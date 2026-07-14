#!/usr/bin/env python3
"""Batch mosaic renderer — render every delivery's mosaic in parallel.

Each delivery's mosaic render (``render_videos.py --mode mosaic``) is a
CPU-bound rasterisation job: 7 camera JPEG decodes + skeleton/chip/text drawing
+ 1920x1080 compositing per frame, with libx264 encode overlapping on spare
cores. On the L40S it runs at ~1.7 CPU-cores and ~100 s for a 600-frame
delivery. The deliveries are fully independent, so the peak lever is
*process-level parallelism across deliveries*, not GPU offload:

* GPU JPEG decode (nvJPEG) was measured 3.2x SLOWER than cv2 here (per-image
  device->host copy + tensor reshape dominate); this launcher forces it OFF
  (``QT_RENDER_GPU_DECODE=0``).
* NVENC is not the bottleneck at 1080p (encode overlaps the draw loop), so no
  GPU encoder is required — libx264 stays.
* Measured scaling on the 8-vCPU L40S: 6 parallel deliveries = 3.2x throughput
  over serial; 8-wide only oversubscribes (load > nproc) with no gain. Default
  ``--jobs`` therefore leaves one core headroom.

The launcher is resumable (skips deliveries whose mosaic mp4 already exists
unless ``--force``), isolates failures per delivery, writes a per-delivery log,
and prints a live + final summary. It shells out to the existing renderer, so
it stays a thin, portable orchestrator (works on the box and the laptop).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RENDERER = ROOT / "src" / "identity" / "visualization" / "render_videos.py"


def discover_deliveries(deliveries_root: Path, run_subdir: str) -> list[tuple[str, Path]]:
    """(delivery_id, run_dir) for every delivery that has renderable predictions."""
    found: list[tuple[str, Path]] = []
    for delivery_dir in sorted(p for p in deliveries_root.iterdir() if p.is_dir()):
        run_dir = delivery_dir / run_subdir
        preds = run_dir / "predictions"
        if not preds.is_dir():
            continue
        if not any(preds.glob("*.jsonl")):
            continue
        found.append((delivery_dir.name, run_dir))
    return found


def mosaic_output_path(run_dir: Path, delivery_id: str) -> Path:
    """Where the renderer writes the mosaic (its default artifact-dir layout)."""
    return run_dir / "visualizations" / "videos" / f"{delivery_id}__all_cameras.mp4"


def _render_one(job: dict) -> dict:
    """Worker: render a single delivery mosaic; never raises (result carries status)."""
    delivery_id = job["delivery_id"]
    run_dir = Path(job["run_dir"])
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    # Force the fast CPU decode path and keep per-process libraries single-threaded
    # so N processes tile cleanly onto N cores instead of oversubscribing.
    env["QT_RENDER_GPU_DECODE"] = "0"
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENCV_NUM_THREADS", "1")
    if job["extra_path"]:
        env["PATH"] = job["extra_path"] + os.pathsep + env.get("PATH", "")

    cmd = [
        job["python"], str(RENDERER),
        "--drive-root", job["drive_root"],
        "--run-dir", str(run_dir),
        "--delivery-id", delivery_id,
        "--mode", "mosaic",
        "--show", job["show"],
    ]
    if job["letterbox_tiles"]:
        cmd.append("--letterbox-tiles")
    if job["max_frames"] is not None:
        cmd += ["--max-frames", str(job["max_frames"])]
    cmd += job["passthrough"]

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env)
    elapsed = time.time() - started
    out_path = mosaic_output_path(run_dir, delivery_id)
    ok = proc.returncode == 0 and out_path.exists()
    return {
        "delivery_id": delivery_id,
        "returncode": proc.returncode,
        "ok": ok,
        "elapsed": elapsed,
        "output": str(out_path),
        "log": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--deliveries-root", required=True,
                   help="Directory holding one subdir per delivery (e.g. ~/pipetrack_v8/deliveries).")
    p.add_argument("--drive-root", required=True,
                   help="Drive-layout root the renderer reads frames/calibration from (e.g. ~/render_drive).")
    p.add_argument("--run-subdir", default="05_global_id",
                   help="Per-delivery run dir holding predictions/ (default: 05_global_id).")
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                   help="Parallel deliveries (default: nproc-1; ~6-7 is the L40S plateau).")
    p.add_argument("--show", default="p4", choices=["p2", "p3", "p4"])
    p.add_argument("--no-letterbox-tiles", dest="letterbox_tiles", action="store_false",
                   help="Disable aspect-correct wide-tile letterboxing (on by default).")
    p.add_argument("--max-frames", type=int, default=None,
                   help="Cap frames per delivery (smoke tests).")
    p.add_argument("--only", nargs="+", default=None,
                   help="Render only these delivery ids.")
    p.add_argument("--force", action="store_true",
                   help="Re-render even if the mosaic mp4 already exists.")
    p.add_argument("--log-dir", default=None,
                   help="Per-delivery logs (default: <deliveries-root>/../mosaic_logs).")
    p.add_argument("--python", default=sys.executable,
                   help="Python interpreter for child renders (default: this one).")
    p.add_argument("--extra-path", default="",
                   help="Prepended to child PATH (e.g. a dir holding a preferred ffmpeg).")
    p.add_argument("--dry-run", action="store_true",
                   help="List what would render and exit.")
    p.add_argument("passthrough", nargs="*",
                   help="Extra args forwarded verbatim to the renderer (after --).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    deliveries_root = Path(args.deliveries_root).expanduser()
    if not deliveries_root.is_dir():
        print(f"error: deliveries-root not found: {deliveries_root}", file=sys.stderr)
        return 2
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else deliveries_root.parent / "mosaic_logs"

    deliveries = discover_deliveries(deliveries_root, args.run_subdir)
    if args.only:
        wanted = set(args.only)
        deliveries = [(d, r) for d, r in deliveries if d in wanted]
        missing = wanted - {d for d, _ in deliveries}
        if missing:
            print(f"warning: requested ids not found: {', '.join(sorted(missing))}", file=sys.stderr)

    todo, skipped = [], []
    for delivery_id, run_dir in deliveries:
        out_path = mosaic_output_path(run_dir, delivery_id)
        if out_path.exists() and not args.force:
            skipped.append(delivery_id)
            continue
        todo.append((delivery_id, run_dir))

    print(f"deliveries discovered: {len(deliveries)} | to render: {len(todo)} | "
          f"already done (skipped): {len(skipped)} | jobs: {args.jobs}")
    if skipped:
        print("  skipping (exists): " + ", ".join(skipped))
    if args.dry_run:
        for delivery_id, run_dir in todo:
            print(f"  would render: {delivery_id}  ({run_dir})")
        return 0
    if not todo:
        print("nothing to render.")
        return 0

    jobs = [{
        "delivery_id": delivery_id,
        "run_dir": str(run_dir),
        "drive_root": args.drive_root,
        "show": args.show,
        "letterbox_tiles": args.letterbox_tiles,
        "max_frames": args.max_frames,
        "python": args.python,
        "extra_path": args.extra_path,
        "passthrough": list(args.passthrough),
        "log_path": str(log_dir / f"{delivery_id}.log"),
    } for delivery_id, run_dir in todo]

    print(f"logs: {log_dir}\n", flush=True)
    results: list[dict] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(_render_one, job): job["delivery_id"] for job in jobs}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            done += 1
            status = "OK " if res["ok"] else "FAIL"
            print(f"[{done}/{len(jobs)}] {status} {res['delivery_id']}  "
                  f"{res['elapsed']:.0f}s  rc={res['returncode']}"
                  + ("" if res["ok"] else f"  (see {res['log']})"), flush=True)

    wall = time.time() - started
    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    print(f"\n=== done in {wall:.0f}s ({wall/60:.1f} min) === "
          f"rendered {len(ok)}/{len(jobs)}  failed {len(failed)}  skipped {len(skipped)}")
    if failed:
        print("FAILURES:")
        for r in sorted(failed, key=lambda r: r["delivery_id"]):
            print(f"  {r['delivery_id']}  rc={r['returncode']}  log={r['log']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
