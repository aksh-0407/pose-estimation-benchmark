"""Unit-testable proxy and ground-truth metrics for P2--P4 tracking.

Proxy metrics are explicitly named as such: they measure geometric consistency
and identity fragmentation, not labelled tracking accuracy. The optional
ground-truth evaluator supplies conventional MOTA/IDF1-style numbers when labels
become available.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.optimize import linear_sum_assignment


def numeric_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    usable = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            usable.append(numeric)
    finite = np.asarray(usable, dtype=float)
    if finite.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "p50": float(np.percentile(finite, 50)),
        "p95": float(np.percentile(finite, 95)),
        "max": float(np.max(finite)),
    }


def association_proxy_metrics(
    correspondence_rows: Iterable[dict[str, Any]],
    *,
    anchor_switch_frames: Iterable[int] = (),
) -> dict[str, Any]:
    rows = list(correspondence_rows)
    clusters = [cluster for row in rows for cluster in row.get("clusters", [])]
    single_count = sum(bool(cluster.get("single_camera")) for cluster in clusters)
    multi = [cluster for cluster in clusters if not cluster.get("single_camera")]
    cycle_count = sum(bool(cluster.get("cycle_consistent")) for cluster in multi)
    anchor_switch_frames = list(anchor_switch_frames)
    return {
        "cluster_count": len(clusters),
        "single_camera_cluster_count": single_count,
        "single_camera_rate": single_count / len(clusters) if clusters else 0.0,
        "cycle_consistency_rate": cycle_count / len(multi) if multi else 1.0,
        "track_confidence": numeric_summary(
            cluster.get("track_confidence") for cluster in clusters
            if cluster.get("track_confidence") is not None
        ),
        "cluster_camera_support": numeric_summary(
            len(cluster.get("members", [])) for cluster in clusters
        ),
        "reprojection_error_px": numeric_summary(
            cluster.get("mean_reprojection_error_px") for cluster in multi
            if cluster.get("mean_reprojection_error_px") is not None
        ),
        "ground_spread_m": numeric_summary(
            cluster.get("ground_spread_m") for cluster in multi
            if cluster.get("ground_spread_m") is not None
        ),
        "anchor_switch_count": len(anchor_switch_frames),
        "anchor_switch_frames": anchor_switch_frames,
        "anchor_switch_frequency": len(anchor_switch_frames) / max(len(rows), 1),
    }


def track_completeness(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Confirmed-frame/span and camera-support summaries per global identity."""

    frames: dict[str, set[int]] = defaultdict(set)
    confirmed: dict[str, set[int]] = defaultdict(set)
    cameras: dict[str, set[str]] = defaultdict(set)
    for record in records:
        frame_index = int(record["frame_index"])
        camera_id = str(record["camera_id"])
        for player in record.get("players", []):
            player_id = player.get("global_player_id")
            if not player_id:
                continue
            frames[player_id].add(frame_index)
            cameras[player_id].add(camera_id)
            if player.get("track_state") == "confirmed":
                confirmed[player_id].add(frame_index)

    per_track = {}
    ratios = []
    for player_id in sorted(frames):
        first, last = min(frames[player_id]), max(frames[player_id])
        span = last - first + 1
        ratio = len(confirmed[player_id]) / span
        ratios.append(ratio)
        per_track[player_id] = {
            "first_frame": first,
            "last_frame": last,
            "span_frames": span,
            "observed_frames": len(frames[player_id]),
            "confirmed_frames": len(confirmed[player_id]),
            "confirmed_frame_completeness": ratio,
            "camera_count": len(cameras[player_id]),
            "cameras": sorted(cameras[player_id]),
        }
    return {
        "track_count": len(per_track),
        "confirmed_frame_completeness": numeric_summary(ratios),
        "per_track": per_track,
    }


def identity_fragmentation_proxy(
    records: Iterable[dict[str, Any]],
    switch_report: Iterable[dict[str, Any]] = (),
) -> dict[str, Any]:
    records = list(records)
    ids = {
        player["global_player_id"]
        for record in records
        for player in record.get("players", [])
        if player.get("global_player_id")
    }
    simultaneous: Counter[int] = Counter()
    for record in records:
        frame = int(record["frame_index"])
        simultaneous[frame] = max(
            simultaneous[frame],
            len({p.get("global_player_id") for p in record.get("players", []) if p.get("global_player_id")}),
        )
    roster_proxy = max(simultaneous.values(), default=0)
    report = list(switch_report)
    return {
        "distinct_global_id_count": len(ids),
        "maximum_simultaneous_id_count_per_camera": roster_proxy,
        "excess_id_fragment_count_proxy": max(0, len(ids) - roster_proxy),
        "stitched_id_switch_proxy_count": len(report),
    }


def identity_collision_metrics(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Count impossible duplicate IDs within a single camera frame."""

    collision_frames = 0
    duplicate_assignments = 0
    examples = []
    for record in records:
        counts = Counter(
            player.get("global_player_id")
            for player in record.get("players", [])
            if player.get("global_player_id")
        )
        duplicates = sorted(player_id for player_id, count in counts.items() if count > 1)
        if not duplicates:
            continue
        collision_frames += 1
        duplicate_assignments += sum(counts[player_id] - 1 for player_id in duplicates)
        if len(examples) < 20:
            examples.append({
                "camera_id": record.get("camera_id"),
                "frame_index": record.get("frame_index"),
                "global_player_ids": duplicates,
            })
    return {
        "same_camera_identity_collision_frames": collision_frames,
        "same_camera_duplicate_identity_assignments": duplicate_assignments,
        "same_camera_identity_collision_examples": examples,
    }


def bbox_iou_xywh(left: Iterable[float], right: Iterable[float]) -> float:
    lx, ly, lw, lh = [float(value) for value in left]
    rx, ry, rw, rh = [float(value) for value in right]
    x1, y1 = max(lx, rx), max(ly, ry)
    x2, y2 = min(lx + lw, rx + rw), min(ly + lh, ry + rh)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = max(0.0, lw * lh) + max(0.0, rw * rh) - intersection
    return intersection / union if union > 0.0 else 0.0


def evaluate_ground_truth(
    prediction_records: Iterable[dict[str, Any]],
    ground_truth_rows: Iterable[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate labelled boxes with deterministic IoU matching.

    Ground-truth rows must contain ``frame_index``, ``camera_id``, ``bbox`` (or
    ``bbox_xywh_px``), and ``gt_id``. IDF1 uses the standard global assignment of
    matched prediction/GT identity pair counts.
    """

    predictions: dict[tuple[int, str], list[tuple[str, list[float]]]] = defaultdict(list)
    for record in prediction_records:
        key = (int(record["frame_index"]), str(record["camera_id"]))
        for player in record.get("players", []):
            if player.get("global_player_id") and player.get("bbox_xywh_px") is not None:
                predictions[key].append((str(player["global_player_id"]), player["bbox_xywh_px"]))

    truth: dict[tuple[int, str], list[tuple[str, list[float]]]] = defaultdict(list)
    for row in ground_truth_rows:
        bbox = row.get("bbox_xywh_px", row.get("bbox"))
        if bbox is None or row.get("gt_id") is None:
            raise ValueError("ground truth rows require gt_id and bbox/bbox_xywh_px")
        truth[(int(row["frame_index"]), str(row["camera_id"]))].append((str(row["gt_id"]), bbox))

    false_positives = false_negatives = identity_switches = matches = 0
    last_prediction_for_gt: dict[tuple[str, str], str] = {}
    identity_pair_counts: Counter[tuple[str, str]] = Counter()
    for key in sorted(set(predictions) | set(truth)):
        pred, gt = predictions[key], truth[key]
        if not pred or not gt:
            false_positives += len(pred)
            false_negatives += len(gt)
            continue
        cost = np.ones((len(gt), len(pred)), dtype=float)
        for gi, (_, gt_bbox) in enumerate(gt):
            for pi, (_, pred_bbox) in enumerate(pred):
                cost[gi, pi] = 1.0 - bbox_iou_xywh(gt_bbox, pred_bbox)
        rows, cols = linear_sum_assignment(cost)
        accepted = [(gi, pi) for gi, pi in zip(rows, cols) if 1.0 - cost[gi, pi] >= iou_threshold]
        matches += len(accepted)
        false_negatives += len(gt) - len(accepted)
        false_positives += len(pred) - len(accepted)
        for gi, pi in accepted:
            gt_id, pred_id = gt[gi][0], pred[pi][0]
            gt_key = (key[1], gt_id)
            previous = last_prediction_for_gt.get(gt_key)
            if previous is not None and previous != pred_id:
                identity_switches += 1
            last_prediction_for_gt[gt_key] = pred_id
            identity_pair_counts[(gt_id, pred_id)] += 1

    gt_total = sum(len(items) for items in truth.values())
    pred_total = sum(len(items) for items in predictions.values())
    gt_ids = sorted({pair[0] for pair in identity_pair_counts})
    pred_ids = sorted({pair[1] for pair in identity_pair_counts})
    idtp = 0
    if gt_ids and pred_ids:
        matrix = np.zeros((len(gt_ids), len(pred_ids)), dtype=float)
        for (gt_id, pred_id), count in identity_pair_counts.items():
            matrix[gt_ids.index(gt_id), pred_ids.index(pred_id)] = count
        assigned_gt, assigned_pred = linear_sum_assignment(-matrix)
        idtp = int(matrix[assigned_gt, assigned_pred].sum())
    idfp, idfn = pred_total - idtp, gt_total - idtp
    denominator = 2 * idtp + idfp + idfn
    return {
        "schema_version": "tracking_ground_truth_metrics/v1",
        "iou_threshold": iou_threshold,
        "ground_truth_detections": gt_total,
        "prediction_detections": pred_total,
        "matches": matches,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "identity_switches": identity_switches,
        "mota": 1.0 - (false_positives + false_negatives + identity_switches) / gt_total
        if gt_total else None,
        "idf1": 2 * idtp / denominator if denominator else None,
        "idtp": idtp,
        "idfp": idfp,
        "idfn": idfn,
    }
