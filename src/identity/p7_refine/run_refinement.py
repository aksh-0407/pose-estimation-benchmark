"""CLI entry point for 07 (refinement) - physics-constrained 3D skeleton refinement.

Runs after identity (06_roles) and before the render/export:

    python -m identity.p7_refine.run_refinement \
        --input-run-dir <06_roles-run> --output-run-dir <07_refine-run> \
        --delivery-id CCPL080626M1_1_14_1 --config configs/07_refine.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from identity.p7_refine.config import load_refine_config  # noqa: E402
from identity.p7_refine.runner import run_refinement  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="07 (refinement) physics-constrained 3D skeleton")
    parser.add_argument("--input-run-dir", required=True, help="identity run dir (predictions/*.jsonl with pose_3d)")
    parser.add_argument("--output-run-dir", required=True, help="refined run dir to write")
    parser.add_argument("--delivery-id", required=True)
    parser.add_argument("--config", default=None, help="optional refine YAML (default: configs/07_refine.yaml)")
    parser.add_argument("--camera", action="append", default=None, help="restrict to camera id(s), e.g. cam_01")
    parser.add_argument("--drive-root", default=None,
                        help="Dataset raw/footage root - provides the calibration for the "
                             "visibility-aware re-lift (without it, re-lift is skipped).")
    parser.add_argument("--expected-frames", type=int, default=600, help="(accepted for CLI uniformity; unused)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    params = load_refine_config(args.config)
    metrics = run_refinement(
        input_run_dir=args.input_run_dir,
        output_run_dir=args.output_run_dir,
        delivery_id=args.delivery_id,
        params=params,
        cameras=args.camera,
        drive_root=args.drive_root,
    )
    print(
        f"refinement: enabled={metrics['enabled']} "
        f"jitter {metrics['jitter_mean_m_before']:.4f} -> {metrics['jitter_mean_m_after']:.4f} m, "
        f"hip {metrics['hip_jitter_mean_m_before']:.4f} -> {metrics['hip_jitter_mean_m_after']:.4f} m, "
        f"bone-CV {metrics['max_bone_cv_before']:.3f} -> {metrics['max_bone_cv_after']:.3f} "
        f"over {metrics['identities']} identities",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
