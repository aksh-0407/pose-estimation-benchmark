"""Deterministic identity colours shared by still and video renderers."""

from __future__ import annotations

import hashlib
import re

IDENTITY_PALETTE: tuple[tuple[int, int, int], ...] = (
    (78, 220, 255), (255, 139, 92), (129, 236, 145), (230, 126, 255),
    (120, 187, 255), (255, 211, 92), (126, 255, 219), (181, 162, 255),
    (90, 164, 255), (255, 116, 166), (105, 238, 205), (196, 231, 105),
)
UNKNOWN_COLOR = (150, 150, 150)


def _palette_index(value: str) -> int:
    canonical = re.fullmatch(r"P(\d+)", value)
    if canonical:
        return (int(canonical.group(1)) - 1) % len(IDENTITY_PALETTE)
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8, person=b"pipetrack").digest()
    return int.from_bytes(digest, "big") % len(IDENTITY_PALETTE)


def color_for_global_id(global_player_id: str | None) -> tuple[int, int, int]:
    if not global_player_id:
        return UNKNOWN_COLOR
    return IDENTITY_PALETTE[_palette_index(str(global_player_id))]


def color_for_player(
    global_player_id: str | None,
    local_track_id: str | None = None,
) -> tuple[int, int, int]:
    if global_player_id:
        return color_for_global_id(global_player_id)
    if local_track_id:
        base = IDENTITY_PALETTE[_palette_index(str(local_track_id))]
        return tuple(int(0.55 * channel + 0.45 * 110) for channel in base)
    return UNKNOWN_COLOR
