from tools.audit_repo import violations


def test_repo_audit_allows_placeholders_and_flags_artifacts():
    paths = [
        "models/rtmpose_l/weights/.gitkeep",
        "models/rtmpose_l/weights/model.pth",
        "models/rtmpose_l/checksums/sha256.json",
        "data/raw/.gitkeep",
        "data/raw/coco/val2017/000000.jpg",
        "data/derived/.gitkeep",
        "data/derived/runs/run_001/deliveries/D/05_global_id/predictions/cam_01.jsonl",
        "data/derived/mosaics/run_001/D/D__all_cameras.mp4",
        "drive/dataset/bt_01/clip/camera01/frame_camera01_000001.jpg",
        "src/main.py",
        "configs/03_association.yaml",
    ]
    # Placeholders (.gitkeep) and tracked source/config are fine; model weights/checksums,
    # local raw inputs, all pipeline outputs under data/derived/, mosaic .mp4, and raw
    # footage are not.
    assert violations(paths) == [
        "data/derived/mosaics/run_001/D/D__all_cameras.mp4",
        "data/derived/runs/run_001/deliveries/D/05_global_id/predictions/cam_01.jsonl",
        "data/raw/coco/val2017/000000.jpg",
        "drive/dataset/bt_01/clip/camera01/frame_camera01_000001.jpg",
        "models/rtmpose_l/checksums/sha256.json",
        "models/rtmpose_l/weights/model.pth",
    ]
