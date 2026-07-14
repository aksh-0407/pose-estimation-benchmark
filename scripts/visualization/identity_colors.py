"""Deterministic identity colours shared by still and video renderers."""

from __future__ import annotations

import hashlib
import re

# 20 maximally-separated colours (W7-RENDER): golden-ratio hue stepping with two
# alternating saturation/value bands, so numerically adjacent global ids (P1, P2, ...)
# land far apart in hue AND neighbouring hues differ in brightness. BGR order.
# Wraparound now every 21st id instead of every 13th; pairwise separation is
# asserted by tests/test_render_labels.py.
IDENTITY_PALETTE: tuple[tuple[int, int, int], ...] = (
    (38, 38, 255), (224, 125, 85), (38, 255, 164), (207, 85, 224),
    (218, 255, 38), (85, 160, 224), (255, 38, 92), (85, 224, 91),
    (110, 38, 255), (224, 172, 85), (38, 255, 237), (224, 85, 195),
    (146, 255, 38), (85, 114, 224), (255, 56, 38), (85, 224, 137),
    (183, 38, 255), (224, 218, 85), (38, 200, 255), (224, 85, 148),
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
