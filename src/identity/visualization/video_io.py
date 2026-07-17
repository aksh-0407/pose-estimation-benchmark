"""Video encoding and frame decoding for the visualization renderers.

``VideoSink`` writes BGR frames to MP4 through the best available encoder
(NVIDIA NVENC when the local ffmpeg exposes it, libx264 otherwise, OpenCV
mp4v as the no-ffmpeg fallback). ``load_image_for_record`` reads a source
frame, using the GPU nvJPEG decoder when available.
"""
from __future__ import annotations

import functools
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@functools.lru_cache(maxsize=1)
def _ffmpeg_has_nvenc() -> bool:
    """True when this ffmpeg build exposes the NVIDIA NVENC H.264 encoder."""
    if not shutil.which("ffmpeg"):
        return False
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10,
        ).stdout
    except Exception:
        return False
    return "h264_nvenc" in out


class VideoSink:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
        use_ffmpeg: bool,
    ) -> None:
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.process: subprocess.Popen | None = None
        self.writer: cv2.VideoWriter | None = None
        self.encoder = "opencv/mp4v"  # actual encoder used (set below); reported in the manifest
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if use_ffmpeg and shutil.which("ffmpeg"):
            common = [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an",
            ]
            if _ffmpeg_has_nvenc():
                # GPU (NVENC) encode: offloads H.264 to the GPU. -cq maps CRF-like
                # quality; p4/hq is a balanced quality preset. If NVENC fails at
                # runtime the encoder errors and close() raises.
                codec = [
                    "-vcodec", "h264_nvenc", "-preset", "p4", "-tune", "hq",
                    "-rc", "vbr", "-cq", str(crf), "-b:v", "0", "-pix_fmt", "yuv420p",
                ]
                self.encoder = "ffmpeg/h264_nvenc"
            else:
                codec = ["-vcodec", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
                self.encoder = "ffmpeg/libx264"
            self.process = subprocess.Popen([*common, *codec, str(path)], stdin=subprocess.PIPE)
        else:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"failed to open video writer: {path}")

    def write(self, frame: np.ndarray) -> None:
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            raise ValueError(
                f"frame shape {frame.shape[:2]} does not match {(self.height, self.width)}"
            )
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        else:
            assert self.writer is not None
            self.writer.write(frame)

    def close(self) -> None:
        if self.process is not None:
            assert self.process.stdin is not None
            self.process.stdin.close()
            return_code = self.process.wait()
            if return_code != 0:
                raise RuntimeError(f"ffmpeg failed for {self.path} with code {return_code}")
        if self.writer is not None:
            self.writer.release()

    def __enter__(self) -> "VideoSink":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        return False


@functools.lru_cache(maxsize=1)
def _gpu_jpeg_decoder():
    """Return (torch, decode_jpeg, read_file) if GPU JPEG decode is usable, else None.

    Offloads the dominant per-frame CPU cost (entropy-decoding 7 large JPEGs) to the
    GPU's nvJPEG engine, leaving the CPU only the drawing/compositing. Set
    ``QT_RENDER_GPU_DECODE=0`` to force the CPU path.
    """
    if os.environ.get("QT_RENDER_GPU_DECODE", "1") == "0":
        return None
    try:
        import torch
        from torchvision.io import decode_jpeg, read_file
        if not torch.cuda.is_available():
            return None
        return (torch, decode_jpeg, read_file)
    except Exception:
        return None


def load_image_for_record(camera_dir: Path, record: dict[str, Any]) -> np.ndarray:
    image_path = camera_dir / record["frame_name"]
    gpu = _gpu_jpeg_decoder()
    if gpu is not None and str(image_path).lower().endswith((".jpg", ".jpeg")):
        try:
            torch, decode_jpeg, read_file = gpu
            data = read_file(str(image_path))
            rgb_chw = decode_jpeg(data, device="cuda")           # (3, H, W) RGB uint8 on GPU
            bgr_hwc = rgb_chw.flip(0).permute(1, 2, 0).contiguous()  # BGR, HWC (cv2 layout)
            return bgr_hwc.cpu().numpy()
        except Exception as exc:
            # Fall back to CPU decode on any GPU/codec hiccup, but say so once so a
            # silently degraded (slower) render is diagnosable.
            if not getattr(_gpu_jpeg_decoder, "_warned", False):
                print(f"WARN: GPU JPEG decode failed ({exc}); falling back to CPU decode", flush=True)
                _gpu_jpeg_decoder._warned = True  # type: ignore[attr-defined]
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    return image


def parse_size(value: str) -> tuple[int, int]:
    import argparse

    try:
        width_text, height_text = value.lower().split("x", 1)
        width, height = int(width_text), int(height_text)
    except Exception as exc:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}, expected WIDTHxHEIGHT") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}, dimensions must be positive")
    return width, height
