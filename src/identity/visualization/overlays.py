"""Drawing primitives and per-tile overlays for the video renderers.

Everything that draws ON a camera frame lives here: text, chips (labels),
skeleton body paint, player boxes and keypoints, ball trails, ghost markers
for undetected players, headers/footers, and the full per-tile compositor
``render_feed_frame``. Panels that are drawn as standalone tiles (roster,
bird's-eye) live in ``panels.py``; artifact loading lives in ``loaders.py``.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import cv2
import numpy as np

from core.keypoints import HALPE26_EDGES
from identity.visualization.identity_colors import (
    color_for_global_id,
    color_for_player,
)

BALL_TRAIL_FRAMES = 18
BALL_COLOR = (0, 215, 255)
EDGE_COLORS = [
    (82, 229, 255),
    (82, 229, 255),
    (254, 189, 96),
    (254, 189, 96),
    (119, 255, 165),
    (118, 205, 255),
    (255, 142, 142),
    (190, 169, 255),
    (118, 205, 255),
    (118, 205, 255),
    (255, 142, 142),
    (255, 142, 142),
    (255, 255, 255),
    (255, 255, 255),
    (209, 209, 209),
    (209, 209, 209),
]

COCO_TORSO = (5, 6, 12, 11)  # L-shoulder, R-shoulder, R-hip, L-hip (quad order)
COCO_HEAD = (0, 1, 2, 3, 4)  # nose, eyes, ears

# Role display names. Currently unused: roles are shown only in the roster
# panel by design (chips stay short); kept as the single place to change the
# on-screen role wording if chips ever carry roles again.
ROLE_TAGS = {
    "bowler": "BOWLER",
    "striker": "STRIKER",
    "non_striker": "NON-STRIKER",
    "wicketkeeper": "KEEPER",
    "umpire": "UMPIRE",
}


def blend_rect(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    alpha: float,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    x1 = max(0, min(image.shape[1], x1))
    x2 = max(0, min(image.shape[1], x2))
    y1 = max(0, min(image.shape[0], y1))
    y2 = max(0, min(image.shape[0], y2))
    if x2 <= x1 or y2 <= y1:
        return
    overlay = image[y1:y2, x1:x2].copy()
    overlay[:] = color
    cv2.addWeighted(overlay, alpha, image[y1:y2, x1:x2], 1.0 - alpha, 0, image[y1:y2, x1:x2])


def draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float,
    color: tuple[int, int, int] = (245, 248, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_chip(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    color: tuple[int, int, int],
    scale: float = 0.48,
) -> tuple[int, int]:
    x, y = origin
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad_x, pad_y = 10, 7
    p1 = (x, y - text_h - pad_y)
    p2 = (x + text_w + 2 * pad_x, y + pad_y)
    # Fill the chip with the player's OWN colour so a label is unmistakably tied to its
    # player even when the leader line is long. The text and border use a contrasting
    # colour (black on light fills, white on dark) picked from the fill's luminance so
    # the text stays readable on any player colour.
    b, g, r = (int(c) for c in color[:3])
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    fg = (0, 0, 0) if luminance > 140 else (255, 255, 255)
    cv2.rectangle(image, p1, p2, color, cv2.FILLED)
    cv2.rectangle(image, p1, p2, fg, 1, cv2.LINE_AA)
    draw_text(image, text, (x + pad_x, y), scale=scale, color=fg, thickness=1)
    return x + text_w + 2 * pad_x + 8, y


def confidence_color(score: float) -> tuple[int, int, int]:
    score = max(0.0, min(1.0, float(score)))
    if score >= 0.75:
        return (111, 255, 151)
    if score >= 0.45:
        return (82, 220, 255)
    return (78, 118, 255)


def scale_player(player: dict[str, Any], *, sx: float, sy: float) -> dict[str, Any]:
    scaled = dict(player)
    x, y, w, h = player["bbox_xywh_px"]
    scaled["bbox_xywh_px"] = [x * sx, y * sy, w * sx, h * sy]
    pose = dict(player["pose_2d"])
    pose["keypoints_px"] = [[point[0] * sx, point[1] * sy] for point in pose["keypoints_px"]]
    scaled["pose_2d"] = pose
    return scaled


def mirror_record(record: dict[str, Any], source_width: int) -> dict[str, Any]:
    """Mirror all player geometry about the vertical axis of the SOURCE frame.

    Overlays (chips, labels, skeletons) are drawn AFTER the image is flipped,
    on mirrored coordinates, so text always renders upright and unmirrored.
    """
    mirrored = dict(record)
    players = []
    for player in record.get("players", []):
        copy = dict(player)
        bbox = player.get("bbox_xywh_px")
        if bbox is not None:
            x, y, w, h = bbox
            copy["bbox_xywh_px"] = [source_width - x - w, y, w, h]
        pose = player.get("pose_2d")
        if pose is not None:
            pose_copy = dict(pose)
            pose_copy["keypoints_px"] = [
                [source_width - point[0], point[1]] for point in pose["keypoints_px"]
            ]
            copy["pose_2d"] = pose_copy
        players.append(copy)
    mirrored["players"] = players
    return mirrored


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2


def place_chip(
    bbox: tuple[int, int, int, int],
    chip_wh: tuple[int, int],
    image_wh: tuple[int, int],
    reserves: tuple[int, int],
    placed: list[tuple[int, int, int, int]],
) -> tuple[int, int, bool]:
    """Collision-aware chip placement.

    Candidates in preference order: above, below, right-of-box, left-of-box, then
    stacked upward above the box (leader line advised). Returns (x, y_baseline,
    stacked) for the first candidate that avoids every previously placed chip and
    the tile's header/footer reserves; falls back to the legacy inside-the-box spot
    when everything collides (still recorded, so later chips dodge it).
    """
    x, y, x2, y2 = bbox
    chip_w, chip_h = chip_wh
    width_limit, height_limit = image_wh
    top_clear, bottom_clear = reserves

    def rect(cx: int, cy_baseline: int) -> tuple[int, int, int, int]:
        return (cx, cy_baseline - chip_h, cx + chip_w, cy_baseline)

    base_x = int(np.clip(x, 6, max(6, width_limit - chip_w - 6)))
    candidates: list[tuple[int, int, bool]] = [
        (base_x, y - 8, False),                          # above
        (base_x, y2 + 10 + chip_h, False),               # below
        (min(x2 + 6, width_limit - chip_w - 6), y + chip_h + 4, False),   # right of box
        (max(6, x - chip_w - 6), y + chip_h + 4, False),                  # left of box
    ]
    for level in range(1, 5):                            # stacked above, rising
        candidates.append((base_x, y - 8 - level * (chip_h + 4), True))
    for level in range(1, 5):                            # stacked below, descending
        candidates.append((base_x, y2 + 10 + chip_h + level * (chip_h + 4), True))
    for level in range(1, 5):                            # stacked beside, right column
        candidates.append(
            (min(x2 + 6, width_limit - chip_w - 6), y + level * (chip_h + 6), True)
        )
    for cx, cy, stacked in candidates:
        if cy - chip_h < top_clear or cy > bottom_clear:
            continue
        r = rect(cx, cy)
        if any(_rects_overlap(r, other) for other in placed):
            continue
        placed.append(r)
        return cx, cy, stacked
    cy = max(top_clear + chip_h, min(y + chip_h + 4, bottom_clear))  # legacy inside
    placed.append(rect(base_x, cy))
    return base_x, cy, False


def draw_body_paint(
    overlay: np.ndarray,
    players: list[dict[str, Any]],
    colors: list[tuple[int, int, int]],
    keypoint_threshold: float,
) -> None:
    """Skeleton body paint: torso quad plus limb capsules plus a head disc,
    filled with the identity colour on a shared overlay layer (blended once by
    the caller). Draws nothing when torso joints are missing; the bbox outline
    still identifies the player."""
    for player, color in zip(players, colors):
        keypoints = player["pose_2d"]["keypoints_px"]
        confidence = player["pose_2d"]["confidence"]
        if len(confidence) < 17:
            continue
        bbox_h = max(1.0, float(player["bbox_xywh_px"][3]))
        limb_w = max(4, int(bbox_h) // 7)

        def pt(i):
            return (int(round(keypoints[i][0])), int(round(keypoints[i][1])))

        torso = [pt(i) for i in COCO_TORSO if float(confidence[i]) >= keypoint_threshold]
        if len(torso) == 4:
            cv2.fillConvexPoly(overlay, np.asarray(torso, dtype=np.int32), color, cv2.LINE_AA)
        for left, right in HALPE26_EDGES:
            if left >= len(confidence) or right >= len(confidence):
                continue
            if min(float(confidence[left]), float(confidence[right])) < keypoint_threshold:
                continue
            cv2.line(overlay, pt(left), pt(right), color, limb_w, cv2.LINE_AA)
        head = [pt(i) for i in COCO_HEAD if float(confidence[i]) >= keypoint_threshold]
        if head:
            cx = int(round(sum(h[0] for h in head) / len(head)))
            cy = int(round(sum(h[1] for h in head) / len(head)))
            cv2.circle(overlay, (cx, cy), max(3, int(bbox_h) // 10), color, -1, cv2.LINE_AA)


def draw_ball_trail(
    image: np.ndarray,
    trail: list[tuple[float, float]],
    *,
    compact: bool,
) -> None:
    if not trail:
        return
    height, width = image.shape[:2]
    points = [(int(round(cx * width)), int(round(cy * height))) for cx, cy in trail]
    line_thickness = 1 if compact else 2
    count = len(points)
    for cursor in range(1, count):
        fade = cursor / count
        color = (0, int(170 + 85 * fade), 255)
        cv2.line(image, points[cursor - 1], points[cursor], color, line_thickness, cv2.LINE_AA)
    cx, cy = points[-1]
    radius = 4 if compact else 6
    cv2.circle(image, (cx, cy), radius + 2, (10, 14, 20), -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, BALL_COLOR, -1, cv2.LINE_AA)
    cv2.circle(image, (cx, cy), radius, (255, 255, 255), 1, cv2.LINE_AA)


class Ghost(NamedTuple):
    """A reprojected marker for a player NOT detected in this camera this frame.

    ``status`` is ``"occluded"`` (still tracked by another camera, so the fused
    position is current) or ``"lost"`` (undetected in every camera, drawn from the
    last-known fused position, so its ground area can be inspected wherever it is
    visible). ``alpha`` in [0, 1] fades the marker as the last observation ages.
    """

    player_id: str
    foot: tuple[float, float]
    status: str = "occluded"
    alpha: float = 1.0


def draw_ghost_markers(
    image: np.ndarray,
    ghosts: list[Ghost],
    *,
    sx: float,
    sy: float,
    avoid_rects: list[tuple[int, int, int, int]] | None = None,
) -> None:
    """Mark players undetected in THIS camera but placed by the tracker elsewhere.

    An ``occluded`` ghost is a player another camera still sees (the detector
    produced nothing here because someone stands in front); a ``lost`` ghost is a
    player no camera sees this frame, drawn from its last-known fused ground position
    so a viewer can see exactly where a disappeared id "should" be in every camera
    that frames that ground. Viewer aid only; no synthetic detection enters the
    pipeline. The chip dodges detection boxes and earlier ghost chips so it never
    covers a visible player.
    """
    height, width = image.shape[:2]
    occupied: list[tuple[int, int, int, int]] = list(avoid_rects or [])

    def overlaps(rect: tuple[int, int, int, int]) -> bool:
        rx1, ry1, rx2, ry2 = rect
        for ox1, oy1, ox2, oy2 in occupied:
            if rx1 < ox2 and rx2 > ox1 and ry1 < oy2 and ry2 > oy1:
                return True
        return False

    def _fade(color: tuple[int, int, int], alpha: float) -> tuple[int, int, int]:
        # Fade toward the dark tile background so an aging ghost visibly dims.
        bg = np.array((10, 14, 21), dtype=float)
        blended = bg + (np.array(color, dtype=float) - bg) * float(np.clip(alpha, 0.15, 1.0))
        return tuple(int(round(v)) for v in blended)

    for ghost in sorted(ghosts):
        player_id, (foot_x, foot_y) = ghost.player_id, ghost.foot
        x, y = int(round(foot_x * sx)), int(round(foot_y * sy))
        if not (0 <= x < width and 0 <= y < height):
            continue
        base = color_for_global_id(player_id)
        lost = ghost.status == "lost"
        # A "lost" ghost is desaturated (greyed) toward its id colour so it reads as
        # "no camera sees this" without losing the id association; occluded stays vivid.
        color = _fade(base if not lost else tuple((c + 150) // 2 for c in base), ghost.alpha)
        ring = (18, 7) if not lost else (20, 8)
        cv2.ellipse(image, (x, y), ring, 0, 0, 360, (12, 16, 24), 3, cv2.LINE_AA)
        cv2.ellipse(image, (x, y), ring, 0, 0, 360, color, 1, cv2.LINE_AA)
        step = 45 if not lost else 30
        for angle in range(0, 360, step):  # dashed inner ring: reads as "not a detection"
            cv2.ellipse(image, (x, y), (11, 4), 0, angle, angle + step // 2, color, 1, cv2.LINE_AA)
        if lost:
            cv2.drawMarker(image, (x, y), color, cv2.MARKER_TILTED_CROSS, 10, 1, cv2.LINE_AA)

        label = f"{player_id}  {ghost.status}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        chip_w, chip_h = text_w + 20, text_h + 14
        candidates = (
            (x - chip_w // 2, y + 14 + text_h),          # below the ellipse
            (x - chip_w // 2, y - 18),                   # above
            (x + 24, y + 4),                             # right
            (x - chip_w - 24, y + 4),                    # left
        )
        chosen = None
        for cx, cy in candidates:
            cx = int(np.clip(cx, 6, max(6, width - chip_w - 6)))
            cy = int(np.clip(cy, 52 + chip_h, height - 34))
            rect = (cx, cy - chip_h, cx + chip_w, cy + 6)
            if not overlaps(rect):
                chosen = (cx, cy, rect)
                break
        if chosen is None:  # everything is crowded: fall back to below, clamped
            cx = int(np.clip(x - chip_w // 2, 6, max(6, width - chip_w - 6)))
            cy = int(np.clip(y + 14 + text_h, 52 + chip_h, height - 34))
            chosen = (cx, cy, (cx, cy - chip_h, cx + chip_w, cy + 6))
        cx, cy, rect = chosen
        occupied.append(rect)
        draw_chip(image, label, (cx, cy), color=color, scale=0.4)


def draw_players(
    image: np.ndarray,
    record: dict[str, Any],
    *,
    sx: float,
    sy: float,
    keypoint_threshold: float,
    compact: bool,
    show: str,
    roles: dict[str, str] | None = None,
    suppressed: set[str] | None = None,
    body_paint: bool = True,
) -> int:
    players = [
        scale_player(player, sx=sx, sy=sy)
        for player in record.get("players", [])
        if player.get("bbox_xywh_px") is not None and player.get("pose_2d") is not None
        and str(player.get("global_player_id")) not in (suppressed or set())
    ]
    line_thickness = 2 if compact else 3
    point_radius = 3 if compact else 5
    font_scale = 0.42 if compact else 0.56

    def _player_color(player):
        if show == "p3" and player.get("_cluster_id") is not None:
            return color_for_global_id(
                f"cluster:{record['frame_index']}:{player['_cluster_id']}"
            )
        return color_for_player(player.get("global_player_id"), player.get("local_track_id"))

    colors = [_player_color(player) for player in players]
    if body_paint and players:
        overlay = image.copy()
        draw_body_paint(overlay, players, colors, keypoint_threshold)
        cv2.addWeighted(overlay, 0.42, image, 0.58, 0, dst=image)
    # Left-to-right processing makes the collision-aware chip placement stable.
    order = sorted(range(len(players)), key=lambda i: players[i]["bbox_xywh_px"][0])
    placed_chips: list[tuple[int, int, int, int]] = []

    for player_index, order_index in enumerate(order, start=1):
        player = players[order_index]
        color = colors[order_index]
        x, y, w, h = [int(round(value)) for value in player["bbox_xywh_px"]]
        x2, y2 = x + w, y + h
        detection_confidence = player.get("detection_confidence")
        if detection_confidence is None:
            detection_confidence = player.get("track_confidence")
        detection_conf = float(detection_confidence or 0.0)

        if not body_paint:
            blend_rect(image, (x, y), (x2, y2), color, 0.12)
        cv2.rectangle(image, (x, y), (x2, y2), color, line_thickness, cv2.LINE_AA)
        corner = max(12, min(w, h) // 7)
        cv2.line(image, (x, y), (x + corner, y), (255, 255, 255), line_thickness, cv2.LINE_AA)
        cv2.line(image, (x, y), (x, y + corner), (255, 255, 255), line_thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y), (x2 - corner, y), (255, 255, 255), line_thickness, cv2.LINE_AA)
        cv2.line(image, (x2, y), (x2, y + corner), (255, 255, 255), line_thickness, cv2.LINE_AA)

        if show == "p4":
            identity = player.get("global_player_id") or "unassigned"
        elif show == "p3":
            identity = f"C{player.get('_cluster_id')}" if player.get("_cluster_id") is not None else "unmatched"
        else:
            identity = player.get("local_track_id") or f"det-{player_index}"
        # Roles are shown ONLY in the roster panel; chips carry identity plus
        # confidence, kept short so side-by-side players stay readable.
        label = f"{identity}  trk {float(player.get('track_confidence') or 0.0):.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        chip_w, chip_h = text_w + 20, text_h + 14
        height_limit, width_limit = image.shape[:2]
        top_clear = 52 if compact else 78
        bottom_clear = height_limit - (32 if compact else 42)
        label_x, label_y, stacked = place_chip(
            (x, y, x2, y2), (chip_w, chip_h), (width_limit, height_limit),
            (top_clear, bottom_clear), placed_chips,
        )
        if stacked:
            # Thicker leader line (matches the box stroke) so a label stacked far from
            # its player is easy to trace back to the right detection box.
            cv2.line(image, (label_x + 6, label_y + 2), (x + 6, y), color,
                     max(2, line_thickness), cv2.LINE_AA)
        next_x, _ = draw_chip(image, label, (label_x, label_y), color=color, scale=font_scale)
        bar_w = 56 if compact else 84
        bar_h = 5 if compact else 7
        bar_x = next_x
        bar_y = label_y - bar_h - 3
        blend_rect(image, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (48, 54, 64), 0.85)
        cv2.rectangle(
            image,
            (bar_x, bar_y),
            (bar_x + int(round(bar_w * detection_conf)), bar_y + bar_h),
            confidence_color(detection_conf),
            -1,
        )

        keypoints = player["pose_2d"]["keypoints_px"]
        confidence = player["pose_2d"]["confidence"]
        for edge_index, (left, right) in enumerate(HALPE26_EDGES):
            if left >= len(confidence) or right >= len(confidence):
                continue
            edge_conf = min(float(confidence[left]), float(confidence[right]))
            if edge_conf < keypoint_threshold:
                continue
            p1 = tuple(int(round(value)) for value in keypoints[left])
            p2 = tuple(int(round(value)) for value in keypoints[right])
            cv2.line(
                image,
                p1,
                p2,
                EDGE_COLORS[edge_index % len(EDGE_COLORS)],
                max(1, line_thickness),
                cv2.LINE_AA,
            )

        for point, score in zip(keypoints, confidence):
            score = float(score)
            if score < keypoint_threshold:
                continue
            center = tuple(int(round(value)) for value in point)
            cv2.circle(image, center, point_radius + 1, (9, 13, 20), -1, cv2.LINE_AA)
            cv2.circle(image, center, point_radius, confidence_color(score), -1, cv2.LINE_AA)
            cv2.circle(image, center, point_radius, (255, 255, 255), 1, cv2.LINE_AA)

    return len(players)


def add_header(
    image: np.ndarray,
    *,
    title: str,
    subtitle: str,
    right_text: str,
    compact: bool,
) -> None:
    height, width = image.shape[:2]
    header_h = 46 if compact else 68
    blend_rect(image, (0, 0), (width, header_h), (11, 16, 24), 0.78)
    cv2.line(image, (0, header_h), (width, header_h), (82, 220, 255), 1, cv2.LINE_AA)
    draw_text(
        image,
        title,
        (16, 29 if compact else 42),
        scale=0.58 if compact else 0.88,
        color=(250, 252, 255),
        thickness=2 if not compact else 1,
    )
    if not compact:
        draw_text(image, subtitle, (18, 61), scale=0.45, color=(184, 196, 214), thickness=1)
    (right_w, _), _ = cv2.getTextSize(right_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42 if compact else 0.55, 1)
    draw_text(
        image,
        right_text,
        (max(16, width - right_w - 18), 29 if compact else 43),
        scale=0.42 if compact else 0.55,
        color=(218, 229, 245),
        thickness=1,
    )


def add_footer(image: np.ndarray, *, compact: bool) -> None:
    height, width = image.shape[:2]
    footer_h = 28 if compact else 38
    blend_rect(image, (0, height - footer_h), (width, height), (11, 16, 24), 0.64)
    text = (
        "Halpe-26  |  bbox + joint confidence"
        if compact
        else "Halpe-26 joints  |  bbox confidence  |  hidden joints suppressed by threshold"
    )
    draw_text(
        image,
        text,
        (16, height - 10),
        scale=0.36 if compact else 0.48,
        color=(198, 208, 222),
        thickness=1,
    )
    legend_x = width - (168 if compact else 230)
    y = height - 13
    for label, color in [("high", (111, 255, 151)), ("mid", (82, 220, 255)), ("low", (78, 118, 255))]:
        cv2.circle(image, (legend_x, y), 4 if compact else 5, color, -1, cv2.LINE_AA)
        draw_text(
            image,
            label,
            (legend_x + 9, y + 4),
            scale=0.34 if compact else 0.42,
            color=(213, 222, 235),
            thickness=1,
        )
        legend_x += 49 if compact else 66


def render_feed_frame(
    image: np.ndarray,
    record: dict[str, Any],
    *,
    output_size: tuple[int, int],
    keypoint_threshold: float,
    compact: bool,
    ball_trail: list[tuple[float, float]] | None = None,
    show: str = "p4",
    mirror: bool = False,
    ghosts: list[Ghost] | None = None,
    letterbox: bool = False,
    roles: dict[str, str] | None = None,
    suppressed: set[str] | None = None,
    body_paint: bool = True,
) -> np.ndarray:
    output_width, output_height = output_size
    source_height, source_width = image.shape[:2]
    if letterbox:
        # A heterogeneous camera (C07 is ~3775x960) stretched into a 16:9 tile
        # distorts the picture. Pad the BOTTOM of the source to the tile aspect
        # before the resize: coordinates are untouched (padding is below every
        # player), the resize becomes aspect-preserving, and overlays stay
        # aligned with the (now undistorted) picture.
        target_height = int(round(source_width * output_height / output_width))
        if target_height > source_height:
            pad = np.zeros((target_height - source_height, source_width, 3), dtype=image.dtype)
            image = np.concatenate([image, pad], axis=0)
            source_height = target_height
    frame = cv2.resize(image, (output_width, output_height), interpolation=cv2.INTER_AREA)
    if mirror:
        # Flip the PICTURE first, then draw every overlay on mirrored
        # coordinates: text and chips always render upright.
        frame = cv2.flip(frame, 1)
        record = mirror_record(record, source_width)
        if ball_trail:
            ball_trail = [(1.0 - cx, cy) for cx, cy in ball_trail]
        if ghosts:
            ghosts = [
                ghost._replace(foot=(source_width - ghost.foot[0], ghost.foot[1]))
                for ghost in ghosts
            ]
    sx = output_width / source_width
    sy = output_height / source_height
    player_count = draw_players(
        frame,
        record,
        sx=sx,
        sy=sy,
        keypoint_threshold=keypoint_threshold,
        compact=compact,
        show=show,
        roles=roles,
        suppressed=suppressed,
        body_paint=body_paint,
    )
    if ghosts:
        avoid_rects = [
            (
                int(player["bbox_xywh_px"][0] * sx),
                int(player["bbox_xywh_px"][1] * sy),
                int((player["bbox_xywh_px"][0] + player["bbox_xywh_px"][2]) * sx),
                int((player["bbox_xywh_px"][1] + player["bbox_xywh_px"][3]) * sy),
            )
            for player in record.get("players", [])
            if player.get("bbox_xywh_px") is not None
        ]
        draw_ghost_markers(frame, ghosts, sx=sx, sy=sy, avoid_rects=avoid_rects)
    if ball_trail:
        draw_ball_trail(frame, ball_trail, compact=compact)
    camera_title = record["camera_id"].replace("_", " ").upper()
    if mirror:
        camera_title += "  (mirrored)"
    if compact:
        title = camera_title
        subtitle = ""
        right = f"players {player_count}"
    else:
        title = f"{camera_title}  |  {record['delivery_id']}"
        subtitle = f"frame {record['frame_index']}  |  source {source_width}x{source_height}"
        right = f"players {player_count}"
    add_header(frame, title=title, subtitle=subtitle, right_text=right, compact=compact)
    add_footer(frame, compact=compact)
    return frame
