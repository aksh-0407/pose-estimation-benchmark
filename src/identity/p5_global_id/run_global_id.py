"""CLI entry point for P4 global identity tracking and stitching."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from identity.p5_global_id.config import load_global_id_config  # noqa: E402
from identity.p5_global_id.runner import run_global_id  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-run-dir", required=True)
    parser.add_argument("--output-run-dir", required=True)
    parser.add_argument("--drive-root", required=True)
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--camera", action="append", default=None)
    parser.add_argument("--expected-frames", type=int, default=600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    metrics = run_global_id(
        input_run_dir=args.input_run_dir,
        output_run_dir=args.output_run_dir,
        drive_root=args.drive_root,
        delivery_id=args.delivery_id,
        config=load_global_id_config(args.config),
        cameras=args.camera,
        expected_frames=args.expected_frames,
    )
    quality = metrics.get("quality_verdict", {})
    print(
        f"P4: {metrics['status']}  frames={metrics['frames_processed']} "
        f"global_ids={metrics['distinct_global_id_count']} merges={metrics['stitched_id_switch_proxy_count']} "
        f"teleports={metrics['teleport_event_count']} "
        f"quality={quality.get('verdict', '?')}"
        + (f" ({'; '.join(quality.get('reasons', []))})" if quality.get("reasons") else ""),
        flush=True,
    )
    return 0 if metrics["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
