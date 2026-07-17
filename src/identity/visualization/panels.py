"""Standalone mosaic tiles: the roster panel, info panel, and bird's-eye view.

These draw whole tiles from scratch (no camera image underneath). Per-frame
overlays drawn on camera pictures live in ``overlays.py``.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from identity.visualization.identity_colors import color_for_global_id
from identity.visualization.loaders import cam_short
from identity.visualization.overlays import blend_rect, draw_text


def draw_roster_panel(
    *,
    size: tuple[int, int],
    roster: list[dict[str, Any]],
    visible_now: dict[str, list[str]],
    accent: tuple[int, int, int] = (129, 236, 145),
) -> np.ndarray:
    """Full-delivery roster: every id ever assigned, who is on screen right now
    (and in which cameras), and the role column filled by the roles stage."""
    width, height = size
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (17, 22, 31)
    blend_rect(panel, (0, 0), (width, height), (28, 36, 48), 0.55)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (58, 68, 82), 1, cv2.LINE_AA)
    cv2.line(panel, (0, 0), (width, 0), accent, 4, cv2.LINE_AA)
    draw_text(panel, "GLOBAL ROSTER", (20, 36), scale=0.62, color=(250, 252, 255), thickness=2)
    live_count = sum(1 for entry in roster if visible_now.get(entry["id"]))
    draw_text(
        panel,
        f"{live_count}/{len(roster)} on screen",
        (width - 150, 34),
        scale=0.42,
        color=(184, 196, 214),
        thickness=1,
    )

    top = 56
    row_height = 24
    rows_per_column = max(1, (height - top - 10) // row_height)
    columns = 1 if len(roster) <= rows_per_column else 2
    column_width = width // columns
    header_color = (140, 152, 170)
    for column in range(columns):
        x0 = 20 + column * column_width
        draw_text(panel, "ID", (x0 + 20, top), scale=0.38, color=header_color, thickness=1)
        draw_text(panel, "LIVE IN", (x0 + 96, top), scale=0.38, color=header_color, thickness=1)
        draw_text(panel, "ROLE", (x0 + column_width - 90, top), scale=0.38, color=header_color, thickness=1)

    for index, entry in enumerate(roster[: rows_per_column * columns]):
        column, row = divmod(index, rows_per_column)
        x0 = 20 + column * column_width
        y = top + 20 + row * row_height
        player_id = entry["id"]
        cams = visible_now.get(player_id) or []
        live = bool(cams)
        color = color_for_global_id(player_id)
        if live:
            blend_rect(panel, (x0 - 8, y - 15), (x0 + column_width - 24, y + 6), (40, 52, 66), 0.55)
        cv2.rectangle(panel, (x0, y - 11), (x0 + 12, y + 1), color, -1, cv2.LINE_AA)
        cv2.rectangle(panel, (x0, y - 11), (x0 + 12, y + 1), (235, 240, 248), 1, cv2.LINE_AA)
        text_color = (245, 248, 255) if live else (120, 132, 150)
        draw_text(panel, player_id, (x0 + 20, y), scale=0.46, color=text_color, thickness=1)
        live_text = " ".join(cam_short(cam) for cam in sorted(cams)) if live else "-"
        draw_text(
            panel,
            live_text,
            (x0 + 96, y),
            scale=0.42,
            color=(148, 236, 164) if live else (96, 106, 122),
            thickness=1,
        )
        draw_text(
            panel,
            entry.get("role") or "-",
            (x0 + column_width - 90, y),
            scale=0.42,
            color=(210, 224, 240),
            thickness=1,
        )
    overflow = len(roster) - rows_per_column * columns
    if overflow > 0:
        draw_text(
            panel,
            f"+{overflow} more",
            (20, height - 12),
            scale=0.4,
            color=(140, 152, 170),
            thickness=1,
        )
    return panel


def draw_info_panel(
    *,
    size: tuple[int, int],
    title: str,
    lines: list[str],
    accent: tuple[int, int, int],
) -> np.ndarray:
    width, height = size
    panel = np.zeros((height, width, 3), dtype=np.uint8)
    panel[:] = (17, 22, 31)
    blend_rect(panel, (0, 0), (width, height), (28, 36, 48), 0.55)
    cv2.rectangle(panel, (0, 0), (width - 1, height - 1), (58, 68, 82), 1, cv2.LINE_AA)
    cv2.line(panel, (0, 0), (width, 0), accent, 4, cv2.LINE_AA)
    draw_text(panel, title, (24, 48), scale=0.72, color=(250, 252, 255), thickness=2)
    y = 88
    for line in lines:
        draw_text(panel, line, (24, y), scale=0.46, color=(204, 214, 230), thickness=1)
        y += 34
    return panel


def draw_bev_panel(
    *,
    size: tuple[int, int],
    frame_grounds: dict[str, tuple[float, float]],
    extents: tuple[float, float, float, float],
    frame_index: int,
    trails: dict[str, list[tuple[int, int]]],
    single_ids: set[str] | None = None,
    ghost_grounds: dict[str, tuple[float, float]] | None = None,
    trail_len: int = 120,
) -> np.ndarray:
    """Bird's-eye mosaic tile: each player's world (x, y) as an id-coloured dot
    plus a short trail.

    ``trails`` is mutated across frames so short motion tails persist.
    ``extents`` is the fixed world window. ``ghost_grounds`` are ids currently
    detected by no camera, drawn as greyed hollow markers at their last-known
    position so a disappearance is visible on the field plot.
    """
    width, height = size
    margin = 30
    header_h = 34
    x_min, x_max, y_min, y_max = extents
    panel = np.full((height, width, 3), (26, 58, 34), dtype=np.uint8)  # deep grass

    # Uniform metric scale (px per world-metre) so the field and its circles are
    # not distorted by the tile aspect ratio.
    world_w = max(x_max - x_min, 1e-6)
    world_h = max(y_max - y_min, 1e-6)
    avail_w = width - 2 * margin
    avail_h = height - 2 * margin - header_h
    scale = min(avail_w / world_w, avail_h / world_h)
    cxw, cyw = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    origin_px = (width / 2.0, (header_h + height) / 2.0)

    def w2p(x: float, y: float) -> tuple[int, int]:
        return (
            int(round(origin_px[0] + (x - cxw) * scale)),
            int(round(origin_px[1] - (y - cyw) * scale)),
        )

    def w_circle(cx: float, cy: float, radius_m: float, color, thickness: int) -> None:
        cv2.circle(panel, w2p(cx, cy), max(1, int(round(radius_m * scale))), color, thickness, cv2.LINE_AA)

    # Subtle mowing stripes for depth.
    for i in range(0, width, max(24, int(6 * scale))):
        shade = 4 if (i // max(24, int(6 * scale))) % 2 == 0 else -4
        panel[header_h:, i:i + max(12, int(3 * scale))] = np.clip(
            panel[header_h:, i:i + max(12, int(3 * scale))].astype(int) + shade, 0, 255
        ).astype(np.uint8)

    # Field reference geometry (metric): 30-yard ring (27.4 m), a nominal
    # boundary arc, and the pitch strip with popping creases in world coordinates.
    w_circle(0.0, 0.0, 68.0, (58, 96, 66), 2)          # nominal boundary
    w_circle(0.0, 0.0, 27.43, (150, 205, 165), 1)      # 30-yard inner ring
    # Pitch strip (3.05 m x 20.12 m along the y axis) as a filled tan rectangle.
    p1, p2 = w2p(-1.525, 10.06), w2p(1.525, -10.06)
    cv2.rectangle(panel, p1, p2, (150, 180, 200), -1, cv2.LINE_AA)
    blend_rect(panel, (min(p1[0], p2[0]), min(p1[1], p2[1])),
               (max(p1[0], p2[0]), max(p1[1], p2[1])), (120, 150, 175), 0.35)
    cv2.rectangle(panel, p1, p2, (210, 225, 235), 1, cv2.LINE_AA)
    for crease_y in (8.84, -8.84):  # popping creases
        cv2.line(panel, w2p(-1.83, crease_y), w2p(1.83, crease_y), (235, 245, 250), 1, cv2.LINE_AA)
    for stump_y in (10.06, -10.06):  # stumps
        cv2.circle(panel, w2p(0.0, stump_y), 2, (245, 250, 255), -1, cv2.LINE_AA)

    # Scale bar (10 m).
    bar = int(round(10.0 * scale))
    bx, by = 16, height - 14
    cv2.line(panel, (bx, by), (bx + bar, by), (220, 230, 240), 2, cv2.LINE_AA)
    draw_text(panel, "10 m", (bx + bar + 6, by + 4), scale=0.34, color=(210, 220, 232))

    live = sorted(frame_grounds)
    for player_id in live:
        x, y = frame_grounds[player_id]
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        pixel = w2p(float(x), float(y))
        trail = trails.setdefault(str(player_id), [])
        trail.append(pixel)
        if len(trail) > trail_len:
            del trail[:-trail_len]
        color = color_for_global_id(str(player_id))
        if len(trail) >= 2:
            cv2.polylines(panel, [np.asarray(trail, np.int32)], False, color, 1, cv2.LINE_AA)
        single = single_ids is not None and str(player_id) in single_ids
        cv2.circle(panel, pixel, 7, (8, 14, 10), -1, cv2.LINE_AA)          # dark halo
        if single:
            cv2.circle(panel, pixel, 5, color, 2, cv2.LINE_AA)             # hollow = 1 camera
        else:
            cv2.circle(panel, pixel, 5, color, -1, cv2.LINE_AA)
            cv2.circle(panel, pixel, 5, (245, 248, 255), 1, cv2.LINE_AA)   # white keyline
        _bev_label(panel, str(player_id), pixel, color)
    # Ghost (disappeared) ids: greyed marker at last-known position, no trail growth.
    for player_id, (x, y) in sorted((ghost_grounds or {}).items()):
        if not (np.isfinite(x) and np.isfinite(y)) or str(player_id) in frame_grounds:
            continue
        pixel = w2p(float(x), float(y))
        base = color_for_global_id(str(player_id))
        grey = tuple(int((c + 150) // 2) for c in base)
        cv2.circle(panel, pixel, 6, (8, 14, 10), -1, cv2.LINE_AA)
        cv2.circle(panel, pixel, 5, grey, 1, cv2.LINE_AA)
        cv2.drawMarker(panel, pixel, grey, cv2.MARKER_TILTED_CROSS, 8, 1, cv2.LINE_AA)
        _bev_label(panel, str(player_id), pixel, grey)

    # Header bar.
    blend_rect(panel, (0, 0), (width, header_h), (12, 20, 16), 0.7)
    cv2.line(panel, (0, 0), (width, 0), (82, 220, 255), 3, cv2.LINE_AA)
    draw_text(panel, "BIRD'S-EYE VIEW", (14, 23), scale=0.56, color=(250, 252, 255), thickness=2)
    ghost_note = f"  ghost {len(ghost_grounds)}" if ghost_grounds else ""
    draw_text(panel, f"f{frame_index}  live {len(live)}{ghost_note}",
              (width - 168, 23), scale=0.42, color=(206, 216, 232))
    return panel


def _bev_label(panel: np.ndarray, text: str, pixel: tuple[int, int], color) -> None:
    """Small readable id label beside a bird's-eye marker (dark keyline for contrast)."""
    x, y = pixel[0] + 8, pixel[1] - 6
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (8, 12, 10), 3, cv2.LINE_AA)
    cv2.putText(panel, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)
