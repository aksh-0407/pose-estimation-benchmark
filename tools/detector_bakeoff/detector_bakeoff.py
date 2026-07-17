#!/usr/bin/env python3
"""Detector-only recall bake-off for Wave 5 (small/distant-player recall).

Runs several detector candidates over a sampled frame set and records raw person
boxes per frame, so recall/precision can be compared per camera before spending
GPU time on full det+pose runs. Pose is untouched (RTMPose-X mandate); only the
person detector feeding it is under test.

Candidates (select with --candidates):
  m640   - RTMDet-m person @ 640 (current production baseline)
  m1280  - RTMDet-m person @ 1280 long-side (2x effective resolution)
  m2560  - RTMDet-m person @ native 2560x1440 (9x cost, the no-seams ceiling)
  t640   - RTMDet-m person @ 640 on a 4x2 overlapping tile grid + full frame,
           merged with cross-tile NMS (SAHI-style)
  l1280  - RTMDet-l (80-class COCO) @ 1280, person class only; weights fetched
           via openmim on first use (skipped with a warning if unavailable)

Dataset layouts are auto-discovered per delivery:
  local : <drive-root>/dataset/bt_0X/<delivery>/camera<NN>/frame_*.jpg
  l40s  : <data-root>/{bt1,bt2,bt3}/<delivery>/camera<NN>/frame_*.jpg

Outputs under --out:
  <candidate>/detections.jsonl  {delivery, camera, frame_index, boxes:[[x1,y1,x2,y2,score],..]}
  <candidate>/summary.json      per-camera counts + throughput
  <candidate>/overlays/*.jpg    a few box-overlay frames per camera for eyeballing
  bakeoff_manifest.json

Boxes are kept down to score 0.10 so the report can study thresholds; the
production cut (0.30) is applied only in summary counts.

Typical L40S sweep (2 deliveries, stride 5, all 7 cams):
  python tools/detector_bakeoff/detector_bakeoff.py \
      --data-root ~/pose_data --out ~/bakeoff_w5 \
      --deliveries CCPL080626M1_1_14_7 CCPL080626M2_1_11_2 --stride 5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]

FRAME_RE = re.compile(r"frame_camera(\d+)_(\d+)\.(jpg|jpeg|png)$")

DET_M_CONFIG = ROOT / "external" / "mmpose" / "demo" / "mmdetection_cfg" / "rtmdet_m_640-8xb32_coco-person.py"
DET_M_CKPT = ROOT / "models" / "rtmdet_m_person" / "weights" / "rtmdet_m_8xb32-100e_coco-obj365-person-235e8209.pth"

SMALL_BOX_PX = 100.0  # box height below this = the "missed band" (frame px)
PRODUCTION_THR = 0.30
KEEP_THR = 0.10


@dataclass
class Candidate:
    name: str
    config: str
    checkpoint: str
    scale: int | None = None          # long-side test scale override (None = config default)
    tiled: bool = False
    tile_grid: tuple[int, int] = (4, 2)
    tile_overlap: float = 0.25
    person_class: int = 0
    batch: int = 8


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None, help="L40S layout root containing bt1/bt2/bt3")
    ap.add_argument("--drive-root", default="data/raw/8_init", help="Dataset raw root containing bt_0X/ (e.g. data/raw/8_init)")
    ap.add_argument("--deliveries", nargs="+", required=True)
    ap.add_argument("--cameras", nargs="+", default=None, help="camera01 / cam_01 / 1 (default all)")
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--frame-limit", type=int, default=None, help="Max frames per camera after stride")
    ap.add_argument("--candidates", nargs="+", default=["m640", "m1280", "m2560", "t640", "l1280"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--overlay-frames", type=int, default=3, help="Overlay jpgs per camera per candidate")
    ap.add_argument("--det-m-config", default=str(DET_M_CONFIG))
    ap.add_argument("--det-m-checkpoint", default=str(DET_M_CKPT))
    return ap.parse_args()


def _norm_cam(cam: str) -> str:
    digits = re.sub(r"\D", "", cam)
    return f"camera{int(digits):02d}"


def discover_frames(args: argparse.Namespace) -> dict[tuple[str, str], list[Path]]:
    """Return {(delivery, cameraNN): [frame paths]} for both supported layouts."""
    roots: list[Path] = []
    if args.data_root:
        base = Path(args.data_root).expanduser()
        roots += sorted(base.glob("bt[0-9]"))
    drive = Path(args.drive_root).expanduser()
    if drive.is_dir():
        roots += sorted(drive.glob("bt_0[0-9]"))
    wanted_cams = {_norm_cam(c) for c in args.cameras} if args.cameras else None
    out: dict[tuple[str, str], list[Path]] = {}
    for root in roots:
        for delivery in args.deliveries:
            ddir = root / delivery
            if not ddir.is_dir():
                continue
            for camdir in sorted(ddir.glob("camera[0-9][0-9]")):
                if wanted_cams and camdir.name not in wanted_cams:
                    continue
                frames = sorted(
                    (p for p in camdir.iterdir() if FRAME_RE.search(p.name)),
                    key=lambda p: int(FRAME_RE.search(p.name).group(2)),
                )
                frames = frames[:: max(1, args.stride)]
                if args.frame_limit:
                    frames = frames[: args.frame_limit]
                if frames:
                    out.setdefault((delivery, camdir.name), []).extend(frames)
    return out


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def _patch_transforms(transforms, scale: int | None) -> None:
    """Patch Resize scale + relax fixed-size Pad in a transform list (in place)."""
    if not transforms:
        return
    for tr in transforms:
        if not isinstance(tr, dict):
            continue
        ttype = tr.get("type", "")
        if scale is not None and "Resize" in ttype and "scale" in tr:
            tr["scale"] = (scale, scale)
        if scale is not None and ttype == "Pad" and "size" in tr:
            del tr["size"]
            tr["size_divisor"] = 32
        # nested wrappers (e.g. MultiScaleFlipAug)
        for key in ("transforms",):
            sub = tr.get(key)
            if isinstance(sub, list):
                if sub and isinstance(sub[0], list):
                    for inner in sub:
                        _patch_transforms(inner, scale)
                else:
                    _patch_transforms(sub, scale)


def build_model(cand: Candidate, device: str):
    from mmdet.apis import init_detector
    from mmengine.config import Config
    from mmpose.utils import adapt_mmdet_pipeline

    cfg = Config.fromfile(cand.config)
    for holder in (cfg.get("test_pipeline"),
                   cfg.get("test_dataloader", {}).get("dataset", {}).get("pipeline")):
        _patch_transforms(holder, cand.scale)
    model = init_detector(cfg, cand.checkpoint, device=device)
    model.cfg = adapt_mmdet_pipeline(model.cfg)
    return model


def resolve_l_checkpoint() -> str | None:
    """Fetch RTMDet-l COCO weights via openmim into models/rtmdet_l_coco/ (once)."""
    dest = ROOT / "models" / "rtmdet_l_coco"
    existing = sorted(dest.glob("*.pth"))
    if existing:
        return str(existing[0])
    try:
        from mim import download
        dest.mkdir(parents=True, exist_ok=True)
        download("mmdet", ["rtmdet_l_8xb32-300e_coco"], dest_root=str(dest))
        existing = sorted(dest.glob("*.pth"))
        return str(existing[0]) if existing else None
    except Exception as exc:  # noqa: BLE001
        print(f"[l1280] skipped: cannot fetch rtmdet_l weights ({exc})", flush=True)
        return None


def make_candidates(args: argparse.Namespace) -> list[Candidate]:
    m_cfg, m_ckpt = args.det_m_config, args.det_m_checkpoint
    table = {
        "m640": Candidate("m640", m_cfg, m_ckpt, scale=None, batch=16),
        "m1280": Candidate("m1280", m_cfg, m_ckpt, scale=1280, batch=8),
        "m2560": Candidate("m2560", m_cfg, m_ckpt, scale=2560, batch=2),
        "t640": Candidate("t640", m_cfg, m_ckpt, scale=None, tiled=True, batch=16),
    }
    out = []
    for name in args.candidates:
        if name == "l1280":
            ckpt = resolve_l_checkpoint()
            if ckpt is None:
                continue
            l_cfg = ROOT / "models" / "rtmdet_l_coco"
            cfgs = sorted(Path(l_cfg).glob("*.py"))
            if not cfgs:
                print("[l1280] skipped: config missing after download", flush=True)
                continue
            out.append(Candidate("l1280", str(cfgs[0]), ckpt, scale=1280, batch=6))
        elif name in table:
            out.append(table[name])
        else:
            sys.exit(f"unknown candidate: {name}")
    return out


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def tile_layout(w: int, h: int, grid: tuple[int, int], overlap: float) -> list[tuple[int, int, int, int]]:
    ncols, nrows = grid
    tw = int(np.ceil(w / ncols * (1.0 + overlap)))
    th = int(np.ceil(h / nrows * (1.0 + overlap)))
    xs = np.linspace(0, w - tw, ncols).round().astype(int) if ncols > 1 else [0]
    ys = np.linspace(0, h - th, nrows).round().astype(int) if nrows > 1 else [0]
    return [(int(x), int(y), tw, th) for y in ys for x in xs]


def nms_xyxy(boxes: np.ndarray, iou_thr: float = 0.6) -> np.ndarray:
    """Greedy NMS; boxes = [x1,y1,x2,y2,score] sorted or not. Returns kept rows."""
    if len(boxes) == 0:
        return boxes
    order = boxes[:, 4].argsort()[::-1]
    boxes = boxes[order]
    keep = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    alive = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        keep.append(i)
        rest = np.where(alive)[0]
        rest = rest[rest > i]
        if rest.size == 0:
            continue
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[rest] - inter + 1e-9)
        alive[rest[iou > iou_thr]] = False
    return boxes[keep]


def extract_person_boxes(det_sample, person_class: int) -> np.ndarray:
    pred = det_sample.pred_instances.cpu().numpy()
    keep = (pred.labels == person_class) & (pred.scores >= KEEP_THR)
    if keep.sum() == 0:
        return np.zeros((0, 5), dtype=np.float32)
    return np.concatenate(
        [pred.bboxes[keep].astype(np.float32), pred.scores[keep, None].astype(np.float32)], axis=1)


def detect_batch(model, images: list[np.ndarray], cand: Candidate) -> list[np.ndarray]:
    from mmdet.apis import inference_detector
    results = inference_detector(model, images)
    if not isinstance(results, list):
        results = [results]
    return [extract_person_boxes(r, cand.person_class) for r in results]


def _drop_tile_clipped(boxes: np.ndarray, tile: tuple[int, int, int, int],
                       frame_wh: tuple[int, int], margin: float = 4.0) -> np.ndarray:
    """Drop boxes touching an INTERIOR tile border - they are partial-person
    fragments; the neighbouring tile or the full-frame pass owns the whole person.
    Borders shared with the frame edge are legitimate truncations and are kept."""
    if len(boxes) == 0:
        return boxes
    x, y, tw, th = tile
    fw, fh = frame_wh
    keep = np.ones(len(boxes), dtype=bool)
    if x > 0:
        keep &= boxes[:, 0] > margin
    if y > 0:
        keep &= boxes[:, 1] > margin
    if x + tw < fw:
        keep &= boxes[:, 2] < tw - margin
    if y + th < fh:
        keep &= boxes[:, 3] < th - margin
    return boxes[keep]


def suppress_contained(boxes: np.ndarray, iom_thr: float = 0.7) -> np.ndarray:
    """Drop a lower-scored box mostly CONTAINED in a higher-scored one (IoM =
    intersection over the smaller area). Kills partial-body fragments that survive
    IoU-NMS because their IoU with the full-person box is small."""
    if len(boxes) <= 1:
        return boxes
    order = boxes[:, 4].argsort()[::-1]
    boxes = boxes[order]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    alive = np.ones(len(boxes), dtype=bool)
    for i in range(len(boxes)):
        if not alive[i]:
            continue
        rest = np.where(alive)[0]
        rest = rest[rest > i]
        if rest.size == 0:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iom = inter / (np.minimum(areas[i], areas[rest]) + 1e-9)
        alive[rest[iom > iom_thr]] = False
    return boxes[alive]


def detect_tiled(model, image: np.ndarray, cand: Candidate) -> np.ndarray:
    h, w = image.shape[:2]
    tiles = tile_layout(w, h, cand.tile_grid, cand.tile_overlap)
    crops = [image[y:y + th, x:x + tw] for (x, y, tw, th) in tiles] + [image]
    results = detect_batch(model, crops, cand)
    all_boxes = []
    for boxes, tile in zip(results[:-1], tiles):
        boxes = _drop_tile_clipped(boxes, tile, (w, h))
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] += tile[0]
            boxes[:, [1, 3]] += tile[1]
            all_boxes.append(boxes)
    if len(results[-1]):
        all_boxes.append(results[-1])  # full-frame pass, already in frame coords
    if not all_boxes:
        return np.zeros((0, 5), dtype=np.float32)
    merged = nms_xyxy(np.concatenate(all_boxes), iou_thr=0.6)
    return suppress_contained(merged, iom_thr=0.7)


def draw_overlay(image: np.ndarray, boxes: np.ndarray, path: Path) -> None:
    import cv2
    img = image.copy()
    for x1, y1, x2, y2, s in boxes:
        col = (0, 220, 0) if s >= PRODUCTION_THR else (0, 165, 255)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), col, 2)
        cv2.putText(img, f"{s:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 82])


@dataclass
class CamStats:
    frames: int = 0
    dets_prod: int = 0        # score >= 0.30
    small_prod: int = 0       # score >= 0.30 and height < SMALL_BOX_PX
    dets_loose: int = 0       # score >= 0.10
    heights: list = field(default_factory=list)


def run_candidate(cand: Candidate, frames_by_cam, out_dir: Path, args) -> dict:
    import cv2
    cv2.setNumThreads(2)
    print(f"[{cand.name}] building model (scale={cand.scale}, tiled={cand.tiled})", flush=True)
    model = build_model(cand, args.device)
    cdir = out_dir / cand.name
    cdir.mkdir(parents=True, exist_ok=True)
    stats: dict[str, CamStats] = {}
    t0 = time.perf_counter()
    nframes = 0
    with open(cdir / "detections.jsonl", "w") as sink:
        for (delivery, cam), frames in sorted(frames_by_cam.items()):
            key = f"{delivery}/{cam}"
            st = stats.setdefault(key, CamStats())
            overlay_idx = set(np.linspace(0, len(frames) - 1, min(args.overlay_frames, len(frames))).astype(int))
            for start in range(0, len(frames), cand.batch if not cand.tiled else 1):
                chunk = frames[start:start + (cand.batch if not cand.tiled else 1)]
                images = [cv2.imread(str(p)) for p in chunk]
                if cand.tiled:
                    boxes_list = [detect_tiled(model, images[0], cand)]
                else:
                    boxes_list = detect_batch(model, images, cand)
                for offset, (path, boxes) in enumerate(zip(chunk, boxes_list)):
                    fidx = int(FRAME_RE.search(path.name).group(2))
                    sink.write(json.dumps({
                        "delivery": delivery, "camera": cam, "frame_index": fidx,
                        "boxes": np.round(boxes, 2).tolist(),
                    }) + "\n")
                    nframes += 1
                    st.frames += 1
                    prod = boxes[boxes[:, 4] >= PRODUCTION_THR] if len(boxes) else boxes
                    st.dets_prod += len(prod)
                    st.dets_loose += len(boxes)
                    if len(prod):
                        hts = prod[:, 3] - prod[:, 1]
                        st.small_prod += int((hts < SMALL_BOX_PX).sum())
                        st.heights += hts.tolist()
                    if (start + offset) in overlay_idx:
                        draw_overlay(images[offset], boxes,
                                     cdir / "overlays" / f"{delivery}__{cam}__f{fidx}.jpg")
    dt = time.perf_counter() - t0
    summary = {
        "candidate": cand.name, "scale": cand.scale, "tiled": cand.tiled,
        "frames": nframes, "wall_s": round(dt, 1), "fps": round(nframes / dt, 2),
        "per_camera": {
            key: {
                "frames": st.frames,
                "dets_per_frame@0.3": round(st.dets_prod / max(1, st.frames), 2),
                "small(<100px)_per_frame@0.3": round(st.small_prod / max(1, st.frames), 3),
                "dets_per_frame@0.1": round(st.dets_loose / max(1, st.frames), 2),
                "median_box_h": round(float(np.median(st.heights)), 1) if st.heights else None,
                "min_box_h": round(float(np.min(st.heights)), 1) if st.heights else None,
            } for key, st in sorted(stats.items())
        },
    }
    (cdir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{cand.name}] done: {nframes} frames in {dt:.0f}s ({nframes/dt:.1f} fps)", flush=True)
    return summary


def main() -> None:
    args = parse_args()
    frames_by_cam = discover_frames(args)
    if not frames_by_cam:
        sys.exit("no frames discovered - check --data-root/--drive-root/--deliveries")
    total = sum(len(v) for v in frames_by_cam.values())
    print(f"discovered {len(frames_by_cam)} camera streams, {total} sampled frames", flush=True)
    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    candidates = make_candidates(args)
    manifest = {"argv": sys.argv[1:], "streams": sorted(f"{d}/{c}" for d, c in frames_by_cam),
                "sampled_frames": total, "summaries": []}
    for cand in candidates:
        try:
            manifest["summaries"].append(run_candidate(cand, frames_by_cam, out_dir, args))
        except Exception as exc:  # noqa: BLE001
            print(f"[{cand.name}] FAILED: {exc}", flush=True)
            manifest["summaries"].append({"candidate": cand.name, "error": str(exc)})
        (out_dir / "bakeoff_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("bake-off complete:", out_dir, flush=True)


if __name__ == "__main__":
    main()
