"""Lightweight IoU micro-track linking.

This is NOT identity tracking (that is P2). It only needs enough short-range temporal
correspondence to give the smoother a per-detection trajectory: greedy IoU association
across consecutive frames with a small gap bridge. It never spans a real occlusion or
crosses cameras, so it cannot introduce identity errors - a mislink just means two
detections are smoothed together for a frame or two.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from identity.p1_stabilization.config import LinkConfig


@dataclass
class _Track:
    last_bbox: tuple[float, float, float, float]  # x, y, w, h
    last_pos: int
    members: list[tuple[int, int]] = field(default_factory=list)  # (frame_pos, player_index)


def _iou_xywh(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def link_micro_tracks(
    frame_boxes: list[list[tuple[int, tuple[float, float, float, float]]]],
    config: LinkConfig,
) -> list[list[tuple[int, int]]]:
    """Link detections into micro-tracks.

    ``frame_boxes[t]`` is the list of ``(player_index, bbox_xywh)`` for frame position ``t``
    (frames already in temporal order). Returns a list of micro-tracks, each a temporally
    ordered list of ``(frame_pos, player_index)``.
    """
    active: list[_Track] = []
    finished: list[_Track] = []

    for pos, detections in enumerate(frame_boxes):
        # Expire tracks whose gap has grown past the bridge limit.
        still_active: list[_Track] = []
        for tr in active:
            if pos - tr.last_pos > config.max_gap_frames:
                finished.append(tr)
            else:
                still_active.append(tr)
        active = still_active

        # Greedy IoU matching: all (track, det) pairs above threshold, highest first.
        pairs: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(active):
            for di, (_pidx, box) in enumerate(detections):
                iou = _iou_xywh(tr.last_bbox, box)
                if iou >= config.iou_min:
                    pairs.append((iou, ti, di))
        pairs.sort(reverse=True)

        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for _iou, ti, di in pairs:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            player_index, box = detections[di]
            active[ti].members.append((pos, player_index))
            active[ti].last_bbox = box
            active[ti].last_pos = pos

        # Unmatched detections spawn new micro-tracks.
        for di, (player_index, box) in enumerate(detections):
            if di in matched_dets:
                continue
            active.append(_Track(last_bbox=box, last_pos=pos, members=[(pos, player_index)]))

    finished.extend(active)
    return [tr.members for tr in finished]
