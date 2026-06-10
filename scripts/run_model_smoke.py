#!/usr/bin/env python3
"""Run one model smoke inference inside its Conda environment."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "model_envs.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--image", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--allow-heavy", action="store_true", help="Allow heavyweight Sapiens2 inference")
    return parser.parse_args()


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expand(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path)))).resolve()


def summarize_mmpose_result(results: Any) -> dict[str, Any]:
    instances = 0
    keypoints = 0
    if results:
        pred = results[0].pred_instances
        if hasattr(pred, "keypoints"):
            instances = int(pred.keypoints.shape[0])
            keypoints = int(pred.keypoints.shape[1]) if pred.keypoints.ndim >= 2 else 0
    return {"instances": instances, "keypoints": keypoints}


def torch_runtime_device(requested_device: str) -> tuple[str, dict[str, Any]]:
    import torch

    metadata = {
        "requested_device": requested_device,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        metadata["runtime_device"] = "cpu"
        metadata["device_fallback_reason"] = "CUDA requested but not visible to torch"
        return "cpu", metadata
    metadata["runtime_device"] = requested_device
    return requested_device, metadata


def smoke_mmpose(model: dict[str, Any], image: Path, device: str) -> dict[str, Any]:
    from mmpose.apis import inference_bottomup, inference_topdown, init_model

    config = expand(model["config"])
    checkpoint = expand(model["checkpoint"])
    runtime_device, device_metadata = torch_runtime_device(device)
    if runtime_device == "cpu" and device.startswith("cuda") and model.get("heavy_runtime"):
        return {
            "status": "ready_runtime_limited",
            "message": "MMPose assets and environment are ready, but CUDA is not visible and this model is marked heavy; CPU smoke inference was skipped.",
            "config": str(config),
            "checkpoint": str(checkpoint),
            **device_metadata,
        }
    cwd = Path.cwd()
    os.chdir(ROOT / "external" / "mmpose")
    try:
        pose_model = init_model(str(config), str(checkpoint), device=runtime_device)
        start = time.perf_counter()
        model_type = pose_model.cfg.model.type
        if model_type == "BottomupPoseEstimator":
            results = inference_bottomup(pose_model, str(image))
        else:
            results = inference_topdown(pose_model, str(image))
        latency_ms = (time.perf_counter() - start) * 1000
        summary = summarize_mmpose_result(results)
        summary.update({"latency_ms": latency_ms, "model_type": model_type, **device_metadata})
        return summary
    finally:
        os.chdir(cwd)


def smoke_dwpose(model: dict[str, Any], image: Path) -> dict[str, Any]:
    import cv2

    sys.path.insert(0, str(ROOT / "external" / "DWPose" / "ControlNet-v1-1-nightly"))
    cwd = Path.cwd()
    os.chdir(ROOT / "external" / "DWPose" / "ControlNet-v1-1-nightly")
    try:
        from annotator.dwpose import DWposeDetector

        detector = DWposeDetector()
        frame = cv2.imread(str(image))
        if frame is None:
            raise RuntimeError(f"Could not read image: {image}")
        start = time.perf_counter()
        output = detector(frame)
        latency_ms = (time.perf_counter() - start) * 1000
        return {"latency_ms": latency_ms, "output_shape": list(output.shape)}
    finally:
        os.chdir(cwd)


def smoke_ultralytics(model: dict[str, Any], image: Path, device: str) -> dict[str, Any]:
    from ultralytics import YOLO

    runtime_device, device_metadata = torch_runtime_device(device)
    yolo_model = YOLO(str(expand(model.get("model_name", model["checkpoint"]))))
    device_arg: str | int = "cpu"
    if runtime_device.startswith("cuda"):
        device_arg = int(runtime_device.split(":", 1)[1]) if ":" in runtime_device else 0
    start = time.perf_counter()
    results = yolo_model(str(image), device=device_arg, verbose=False)
    latency_ms = (time.perf_counter() - start) * 1000
    keypoints = 0
    instances = 0
    if results and results[0].keypoints is not None:
        instances = int(len(results[0].keypoints))
        keypoints = int(results[0].keypoints.data.shape[1]) if results[0].keypoints.data.ndim >= 2 else 0
    return {"latency_ms": latency_ms, "instances": instances, "keypoints": keypoints, **device_metadata}


def smoke_mediapipe(model: dict[str, Any], image: Path) -> dict[str, Any]:
    import cv2
    import mediapipe as mp

    if not hasattr(mp, "solutions"):
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        base_options = python.BaseOptions(model_asset_path=str(expand(model["checkpoint"])))
        options = vision.PoseLandmarkerOptions(base_options=base_options)
        mp_image = mp.Image.create_from_file(str(image))
        start = time.perf_counter()
        with vision.PoseLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
        latency_ms = (time.perf_counter() - start) * 1000
        landmarks = 0
        instances = len(result.pose_landmarks)
        if result.pose_landmarks:
            landmarks = len(result.pose_landmarks[0])
        return {"latency_ms": latency_ms, "instances": instances, "keypoints": landmarks}

    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image}")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()
    with mp.solutions.pose.Pose(static_image_mode=True, model_complexity=2) as pose:
        result = pose.process(rgb)
    latency_ms = (time.perf_counter() - start) * 1000
    landmarks = 0 if result.pose_landmarks is None else len(result.pose_landmarks.landmark)
    return {"latency_ms": latency_ms, "instances": 1 if landmarks else 0, "keypoints": landmarks}


def smoke_vitpose(model: dict[str, Any], image: Path, device: str) -> dict[str, Any]:
    import cv2
    from mmpose.apis import inference_top_down_pose_model, init_pose_model

    checkpoint = expand(model["checkpoint"])
    pose_model = init_pose_model(str(expand(model["config"])), str(checkpoint), device=device)
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError(f"Could not read image: {image}")
    h, w = frame.shape[:2]
    person_results = [{"bbox": [0, 0, w, h]}]
    start = time.perf_counter()
    pose_results, _ = inference_top_down_pose_model(
        pose_model,
        str(image),
        person_results,
        bbox_thr=None,
        format="xyxy",
    )
    latency_ms = (time.perf_counter() - start) * 1000
    keypoints = 0
    if pose_results and "keypoints" in pose_results[0]:
        keypoints = int(pose_results[0]["keypoints"].shape[0])
    return {"latency_ms": latency_ms, "instances": len(pose_results), "keypoints": keypoints}


def sapiens_detector_path(model: dict[str, Any]) -> Path | None:
    if model.get("detector"):
        return expand(model["detector"])
    for asset in model.get("assets", []):
        if asset.get("kind") == "hf_repo":
            return expand(asset["path"])
    return None


def smoke_sapiens2(model: dict[str, Any], image: Path, device: str, allow_heavy: bool) -> dict[str, Any]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib"))
    import sapiens  # noqa: F401
    import torch

    checkpoint = expand(model["checkpoint"])
    config = expand(model["config"])
    detector = sapiens_detector_path(model)
    metadata = {
        "checkpoint": str(checkpoint),
        "config": str(config),
        "detector": str(detector) if detector else None,
        "torch_version": torch.__version__,
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "requested_device": device,
    }

    missing = []
    if not checkpoint.exists():
        missing.append(str(checkpoint))
    if not config.exists():
        missing.append(str(config))
    if detector is None or not detector.exists():
        missing.append(str(detector) if detector else "detector path not configured")
    if not image.exists():
        missing.append(str(image))
    if missing:
        return {
            "status": "missing_assets",
            "missing_assets": missing,
            **metadata,
        }

    if not allow_heavy:
        return {
            "status": "ready_heavy_skipped",
            "message": "Sapiens2 import, config, detector, and checkpoint checks passed; full 1B inference skipped without --allow-heavy.",
            **metadata,
        }

    if device.startswith("cuda") and not torch.cuda.is_available():
        return {
            "status": "ready_runtime_limited",
            "message": "Sapiens2 assets and Python package are ready, but this Conda environment cannot see CUDA. Full 1B inference was not launched.",
            **metadata,
        }

    script = ROOT / "external" / "sapiens2" / "sapiens" / "pose" / "tools" / "vis" / "vis_pose.py"
    workdir = ROOT / "external" / "sapiens2" / "sapiens" / "pose"
    with tempfile.TemporaryDirectory(prefix="sapiens2_smoke_") as temp_dir:
        temp_root = Path(temp_dir)
        input_dir = temp_root / "input"
        output_dir = temp_root / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        image_copy = input_dir / image.name
        shutil.copy2(image, image_copy)
        command = [
            sys.executable,
            str(script),
            str(detector),
            str(config),
            str(checkpoint),
            "--input",
            str(input_dir),
            "--output",
            str(output_dir),
            "--device",
            device,
            "--predictions-name",
            "smoke_predictions.json",
        ]
        env = {
            **os.environ,
            "MPLCONFIGDIR": str(Path(tempfile.gettempdir()) / "matplotlib"),
            "PYTHONNOUSERSITE": "1",
        }
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            timeout=900,
            env=env,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        predictions = output_dir / "smoke_predictions.json"
        if completed.returncode != 0:
            return {
                "status": "error",
                "latency_ms": latency_ms,
                "error": completed.stdout[-3000:],
                **metadata,
            }
        records = []
        if predictions.exists():
            records = json.loads(predictions.read_text(encoding="utf-8"))
        instances = 0
        if records:
            pose_results = records[0].get("pose_results", [])
            instances = len(pose_results)
        return {
            "status": "ok",
            "latency_ms": latency_ms,
            "instances": instances,
            "keypoints": 308,
            "predictions_written": predictions.exists(),
            **metadata,
        }


def smoke_openpose(model: dict[str, Any], image: Path) -> dict[str, Any]:
    executable = model.get("executable", "openpose")
    resolved = shutil.which(executable) or (str(expand(executable)) if Path(str(executable)).exists() else None)
    if resolved is None:
        return {
            "status": "missing_runtime",
            "message": f"OpenPose executable not found: {executable}",
        }
    model_folder = expand(model["checkpoint"]).parents[2]
    with tempfile.TemporaryDirectory(prefix="openpose_smoke_") as temp_dir:
        temp_root = Path(temp_dir)
        input_dir = temp_root / "input"
        output_dir = temp_root / "output"
        input_dir.mkdir()
        output_dir.mkdir()
        shutil.copy2(image, input_dir / image.name)
        command = [
            resolved,
            "--image_dir",
            str(input_dir),
            "--write_json",
            str(output_dir),
            "--display",
            "0",
            "--render_pose",
            "0",
            "--net_resolution",
            "-1x160",
            "--model_pose",
            "BODY_25",
            "--model_folder",
            str(model_folder),
        ]
        start = time.perf_counter()
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        latency_ms = (time.perf_counter() - start) * 1000
        json_files = sorted(output_dir.glob("*_keypoints.json"))
        if completed.returncode != 0:
            return {
                "status": "error",
                "latency_ms": latency_ms,
                "error": completed.stdout[-2000:],
            }
        instances = 0
        keypoints = 25
        if json_files:
            payload = json.loads(json_files[0].read_text(encoding="utf-8"))
            instances = len(payload.get("people", []))
        return {"latency_ms": latency_ms, "instances": instances, "keypoints": keypoints}


def missing_required_assets(model: dict[str, Any]) -> list[str]:
    missing = []
    for asset in model.get("assets", []):
        if asset.get("required_for_smoke") and not expand(asset["path"]).exists():
            missing.append(str(expand(asset["path"])))
    for asset in model.get("manual_assets", []):
        if asset.get("required_for_smoke") and not expand(asset["path"]).exists():
            missing.append(str(expand(asset["path"])))
    checkpoint = model.get("checkpoint")
    if checkpoint and checkpoint not in {"bundled"} and checkpoint.endswith((".pth", ".onnx", ".safetensors", ".task")):
        path = expand(checkpoint)
        if not path.exists() and str(path) not in missing:
            missing.append(str(path))
    return missing


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    model = config["models"][args.model]
    defaults = config.get("defaults", {})
    image = expand(args.image or model.get("smoke_image") or defaults["smoke_image"])

    started = time.perf_counter()
    result: dict[str, Any] = {
        "model_id": args.model,
        "env_name": model["env_name"],
        "smoke_profile": model["smoke_profile"],
        "image": str(image),
        "status": "ok",
    }

    missing = missing_required_assets(model)
    if missing:
        result.update({"status": "missing_assets", "missing_assets": missing})
        print(json.dumps(result, indent=2, sort_keys=True))
        return 2

    try:
        profile = model["smoke_profile"]
        if profile == "mmpose":
            details = smoke_mmpose(model, image, args.device)
        elif profile == "dwpose":
            details = smoke_dwpose(model, image)
        elif profile == "ultralytics":
            details = smoke_ultralytics(model, image, args.device)
        elif profile == "mediapipe":
            details = smoke_mediapipe(model, image)
        elif profile == "vitpose":
            details = smoke_vitpose(model, image, args.device)
        elif profile == "sapiens2":
            details = smoke_sapiens2(model, image, args.device, args.allow_heavy)
        elif profile == "openpose":
            details = smoke_openpose(model, image)
        else:
            raise RuntimeError(f"Unknown smoke profile: {profile}")
        result.update(details)
        result["elapsed_ms"] = (time.perf_counter() - started) * 1000
        if details.get("status"):
            result["status"] = details["status"]
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_ms": (time.perf_counter() - started) * 1000,
            }
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
