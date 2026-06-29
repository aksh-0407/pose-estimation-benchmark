from scripts.visualization.identity_colors import color_for_global_id, color_for_player


def test_global_identity_color_is_stable_and_sequential_ids_are_distinct():
    assert color_for_global_id("P001") == color_for_global_id("P001")
    assert color_for_global_id("P001") != color_for_global_id("P002")


def test_local_fallback_is_deterministic_and_muted():
    assert color_for_player(None, "cam_01_trk_0001") == color_for_player(None, "cam_01_trk_0001")
    assert color_for_player(None, None) == (150, 150, 150)
