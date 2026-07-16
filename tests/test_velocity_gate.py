from __future__ import annotations

from identity.p5_global_id.runner import _velocity_gate_ground_rows

FPS = 50.0


def _rows(series: dict[str, dict[int, tuple[float, float]]]) -> dict[int, list[dict]]:
    """series[gid][frame] = (x, y) -> ground_rows_by_frame."""
    by_frame: dict[int, list[dict]] = {}
    frames = sorted({f for per in series.values() for f in per})
    for f in frames:
        rows = []
        for gid, per in series.items():
            if f in per:
                rows.append({"global_player_id": gid, "ground_xy": list(per[f])})
        by_frame[f] = rows
    return by_frame


def _frames_for(out: dict[int, list[dict]], gid: str) -> set[int]:
    return {f for f, rows in out.items() if any(r["global_player_id"] == gid for r in rows)}


def test_single_teleport_frame_is_dropped_neighbours_kept():
    # frame 2 jumps 24 m in one 1/50 s step (>1000 m/s) then returns -> drop only frame 2
    rows = _rows({"P1": {0: (0.0, 0.0), 1: (0.1, 0.0), 2: (24.0, 0.0), 3: (0.2, 0.0)}})
    out, n = _velocity_gate_ground_rows(rows, frame_rate_fps=FPS, max_mps=12.0, max_consec_drops=5)
    assert n == 1
    assert _frames_for(out, "P1") == {0, 1, 3}


def test_reanchor_after_max_consecutive_drops():
    # sustained jump to (100,0) -> drop up to max_consec_drops (5) then RE-ANCHOR and keep the rest
    series = {"P2": {0: (0.0, 0.0)}}
    for f in range(1, 8):
        series["P2"][f] = (100.0, 0.0)
    out, n = _velocity_gate_ground_rows(_rows(series), frame_rate_fps=FPS, max_mps=12.0, max_consec_drops=5)
    assert n == 5
    assert _frames_for(out, "P2") == {0, 6, 7}  # first kept, 1-5 dropped, re-anchor at 6


def test_no_teleport_is_byte_identical_passthrough():
    # slow walk (<=12 m/s everywhere) -> unchanged object, zero drops
    rows = _rows({"P3": {f: (0.1 * f, 0.0) for f in range(10)}})  # 0.1 m/frame = 5 m/s
    out, n = _velocity_gate_ground_rows(rows, frame_rate_fps=FPS, max_mps=12.0, max_consec_drops=5)
    assert n == 0
    assert out is rows  # same object returned when nothing is dropped


def test_gap_scaled_allowance_keeps_reacquired_track():
    # id absent frames 1-9, reappears at frame 10 having moved 2 m over 10 frames = 10 m/s -> KEEP
    rows = _rows({"P4": {0: (0.0, 0.0), 10: (2.0, 0.0)}})
    out, n = _velocity_gate_ground_rows(rows, frame_rate_fps=FPS, max_mps=12.0, max_consec_drops=5)
    assert n == 0
    assert _frames_for(out, "P4") == {0, 10}


def test_gate_is_per_id_independent():
    # P5 teleports at frame 1; P6 is clean -> only P5 frame 1 dropped, P6 untouched
    rows = _rows({
        "P5": {0: (0.0, 0.0), 1: (50.0, 0.0), 2: (0.0, 0.0)},
        "P6": {0: (5.0, 5.0), 1: (5.1, 5.0), 2: (5.2, 5.0)},
    })
    out, n = _velocity_gate_ground_rows(rows, frame_rate_fps=FPS, max_mps=12.0, max_consec_drops=5)
    assert n == 1
    assert _frames_for(out, "P5") == {0, 2}
    assert _frames_for(out, "P6") == {0, 1, 2}
