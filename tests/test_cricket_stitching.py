from __future__ import annotations

from dataclasses import replace

import numpy as np

from scripts.global_id.config import P4BConfig, P4Config
from scripts.global_id.stitching import (
    Segment,
    build_link_costs,
    extract_segments,
    remap_ids,
    solve_flow,
    velocity_continuity_cost,
)


def _segment(sid, player_id, start, end, first, last, role="unknown", velocity=(0.0, 0.0)):
    return Segment(
        sid, player_id, start, end, np.asarray(first, float), np.asarray(last, float),
        role, np.asarray(velocity, float),
    )


def test_extract_segments_deduplicates_camera_records_and_splits_gaps():
    records = []
    ground = {}
    for camera in ("cam_01", "cam_02"):
        for frame in (1, 2, 4):
            records.append({
                "frame_index": frame,
                "camera_id": camera,
                "players": [{"global_player_id": "P001", "track_state": "confirmed", "role": "fielder"}],
            })
            ground[("P001", frame)] = np.array([frame * 0.1, 0.0])
    segments = extract_segments(records, ground)
    assert [(item.start_frame, item.end_frame) for item in segments] == [(1, 2), (4, 4)]


def test_flow_links_feasible_nearby_segments_and_rejects_impossible_link():
    config = P4Config(p4b=replace(P4BConfig(), new_traj_cost_factor=2.0))
    segments = [
        _segment(0, "P001", 0, 10, (0, 0), (0, 0)),
        _segment(1, "P002", 12, 20, (0.1, 0), (1, 0)),
        _segment(2, "P003", 12, 20, (50, 0), (51, 0)),
    ]
    edges = build_link_costs(segments, config)
    links = solve_flow(segments, edges, config)
    assert links == {0: 1}


def test_role_and_velocity_costs_disfavor_inconsistent_links():
    aligned = _segment(0, "P001", 0, 10, (0, 0), (0, 0), "bowler", (1, 0))
    forward = _segment(1, "P002", 12, 20, (0.2, 0), (0.4, 0), "bowler")
    backward = _segment(2, "P003", 12, 20, (-0.2, 0), (-0.4, 0), "wicketkeeper")
    assert velocity_continuity_cost(aligned, forward) < velocity_continuity_cost(aligned, backward)
    edges = build_link_costs([aligned, forward, backward], P4Config())
    costs = {(edge.source_seg_id, edge.target_seg_id): edge.cost for edge in edges}
    assert costs[(0, 1)] < costs[(0, 2)]


def test_remap_ids_uses_earliest_chain_identity():
    segments = [
        _segment(0, "P010", 0, 3, (0, 0), (0, 0)),
        _segment(1, "P002", 5, 7, (0, 0), (0, 0)),
    ]
    records = [{
        "frame_index": 5,
        "players": [{"global_player_id": "P002"}],
    }]
    report = remap_ids(records, segments, {0: 1})
    assert records[0]["players"][0]["global_player_id"] == "P010"
    assert report == [{"merged_id": "P002", "into_id": "P010", "at_frame": 5}]


def test_remap_ids_rejects_same_camera_frame_identity_collision():
    segments = [
        _segment(0, "P001", 0, 3, (0, 0), (0, 0)),
        _segment(1, "P002", 5, 7, (0, 0), (0, 0)),
    ]
    records = [{
        "frame_index": 2,
        "camera_id": "cam_01",
        "players": [
            {"global_player_id": "P001"},
            {"global_player_id": "P002"},
        ],
    }]
    assert remap_ids(records, segments, {0: 1}) == []
    assert [player["global_player_id"] for player in records[0]["players"]] == ["P001", "P002"]
