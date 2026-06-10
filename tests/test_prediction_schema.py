import pytest

from pose_estimation.adapters.base import AdapterResult, build_prediction_record
from pose_estimation.predictions import SCHEMA_VERSION, validate_prediction_record


def test_prediction_record_contains_native_and_coco17_targets():
    result = AdapterResult(
        keypoints=[[float(i), float(i + 1), 0.9] for i in range(17)],
        source_skeleton="coco_17",
        bbox_xyxy=[0.0, 1.0, 2.0, 3.0],
        score=0.8,
        timing_ms={"inference": 1.2},
        metadata={"source": "unit"},
    )

    record = build_prediction_record(
        run_id="run",
        model_id="model",
        dataset_id="dataset",
        sample_id="sample",
        image_id="image",
        person_id="person",
        result=result,
    ).to_dict()

    validate_prediction_record(record)
    assert record["schema_version"] == SCHEMA_VERSION
    assert set(record["target_skeletons"]) == {"native", "coco_17"}


def test_prediction_validation_rejects_missing_required_fields():
    with pytest.raises(ValueError):
        validate_prediction_record({"schema_version": SCHEMA_VERSION})

