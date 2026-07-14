#!/usr/bin/env python3
"""Data-parallel Phase-1 launcher — several GPU-sharing inference processes.

A single ``run_phase1_l40s.py`` process is GPU-compute-bound at ~28 f/s on the
L40S while using only ~1.6 GB of the 46 GB VRAM (measured by ``--sweep --grid``):
the RTMDet+RTMPose forward passes do not saturate the SMs, and larger batch sizes
do NOT raise throughput (batch-invariant, flat f/s). The idle head-room is
extracted by running several inference processes concurrently on the same GPU,
each on a disjoint slice of deliveries:

    1 proc  = 28.4 f/s     2 procs = 42.6 f/s (1.5x)     3 procs = 56.6 f/s (2.0x)

Beyond ~3 the 8 vCPUs' JPEG decode becomes the ceiling, so this defaults to 3
shards with reduced per-process io-workers so the aggregate decode threads stay
near the core count. Each shard writes per-camera prediction files into the SAME
``--output-dir`` (filenames are ``<group>__<delivery>__cam_NN.jsonl`` — disjoint
across shards, no collision) and is ``--resume`` safe.

Sharding is by whole delivery so a delivery's 7 cameras stay in one process
(calibration/model warm-load amortised). Deliveries are round-robined across
shards for even load. Everything else (model, tiled-det, nms, batch sizes) is
forwarded verbatim to ``run_phase1_l40s.py``.

Example (production, 3-wide):
    python scripts/inference/run_phase1_parallel.py --shards 3 \
        --pose-data ~/pose_data --output-dir ~/pipetrack_v8/p1_rtmpose-x-tiled \
        -- --tiled-det --nms-thr 0.55 --det-batch-size 8 --pose-batch-size 512 \
           --io-workers 4
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "inference" / "run_phase1_l40s.py"
GROUP_RE = re.compile(r"^bt_?0*(?P<num>\d+)$", re.IGNORECASE)


def discover_deliveries(pose_data: Path) -> list[str]:
    """Every delivery id present under the bt*/ group dirs, sorted, de-duplicated."""
    deliveries: set[str] = set()
    for group_dir in sorted(p for p in pose_data.iterdir() if p.is_dir()):
        if not GROUP_RE.match(group_dir.name):
            continue
        for delivery_dir in sorted(p for p in group_dir.iterdir() if p.is_dir()):
            if any(c.name.startswith("camera") for c in delivery_dir.iterdir() if c.is_dir()):
                deliveries.add(delivery_dir.name)
    return sorted(deliveries)


def round_robin(items: list[str], shards: int) -> list[list[str]]:
    buckets: list[list[str]] = [[] for _ in range(shards)]
    for i, item in enumerate(items):
        buckets[i % shards].append(item)
    return [b for b in buckets if b]


def _run_shard(job: dict) -> dict:
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        job["python"], str(RUNNER),
        "--pose-data", job["pose_data"],
        "--output-dir", job["output_dir"],
        "--deliveries", *job["deliveries"],
        *job["passthrough"],
    ]
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("CMD: " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    return {
        "shard": job["shard"],
        "deliveries": job["deliveries"],
        "returncode": proc.returncode,
        "ok": proc.returncode == 0,
        "elapsed": time.time() - started,
        "log": str(log_path),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pose-data", default="/home/ubuntu/pose_data",
                   help="Root holding bt1/bt2/bt3 (default: /home/ubuntu/pose_data).")
    p.add_argument("--output-dir", required=True,
                   help="Shared prediction output dir (all shards write here).")
    p.add_argument("--shards", type=int, default=3,
                   help="Concurrent GPU-sharing processes (default: 3, the L40S ~2x plateau).")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--log-dir", default=None,
                   help="Per-shard logs (default: <output-dir>/parallel_logs).")
    p.add_argument("--only", nargs="+", default=None,
                   help="Restrict to these delivery ids before sharding.")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("passthrough", nargs="*",
                   help="Args after -- forwarded to run_phase1_l40s.py "
                        "(e.g. --tiled-det --nms-thr 0.55 --det-batch-size 8 ...).")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    pose_data = Path(args.pose_data).expanduser()
    if not pose_data.is_dir():
        print(f"error: pose-data not found: {pose_data}", file=sys.stderr)
        return 2
    deliveries = discover_deliveries(pose_data)
    if args.only:
        wanted = set(args.only)
        deliveries = [d for d in deliveries if d in wanted]
    if not deliveries:
        print("no deliveries discovered.", file=sys.stderr)
        return 2

    shards = max(1, args.shards)
    buckets = round_robin(deliveries, shards)
    log_dir = Path(args.log_dir).expanduser() if args.log_dir else Path(args.output_dir).expanduser() / "parallel_logs"

    print(f"deliveries: {len(deliveries)} | shards: {len(buckets)} | output: {args.output_dir}")
    for i, bucket in enumerate(buckets):
        print(f"  shard {i}: {len(bucket)} deliveries -> {', '.join(bucket)}")
    if args.dry_run:
        return 0

    jobs = [{
        "shard": i,
        "deliveries": bucket,
        "python": args.python,
        "pose_data": str(pose_data),
        "output_dir": args.output_dir,
        "passthrough": list(args.passthrough),
        "log_path": str(log_dir / f"shard_{i}.log"),
    } for i, bucket in enumerate(buckets)]

    print(f"logs: {log_dir}\n", flush=True)
    started = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {pool.submit(_run_shard, job): job["shard"] for job in jobs}
        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            status = "OK " if res["ok"] else "FAIL"
            print(f"[shard {res['shard']}] {status} {res['elapsed']:.0f}s rc={res['returncode']}"
                  + ("" if res["ok"] else f"  (see {res['log']})"), flush=True)

    wall = time.time() - started
    failed = [r for r in results if not r["ok"]]
    print(f"\n=== done in {wall:.0f}s ({wall/60:.1f} min) === "
          f"shards {len(results)-len(failed)}/{len(results)} ok, {len(failed)} failed")
    if failed:
        for r in sorted(failed, key=lambda r: r["shard"]):
            print(f"  shard {r['shard']}  rc={r['returncode']}  log={r['log']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
