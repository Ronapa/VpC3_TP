import json

import numpy as np

from grape_autoseg.pipeline.coco import mask_to_rle, prediction_to_coco, rle_to_mask, xyxy_to_xywh


def _ring_mask() -> np.ndarray:
    mask = np.zeros((40, 60), dtype=bool)
    mask[5:30, 10:50] = True
    mask[12:20, 20:35] = False  # agujero
    return mask


def test_rle_roundtrip():
    mask = _ring_mask()
    rle = mask_to_rle(mask)
    assert rle["size"] == [40, 60]
    assert isinstance(rle["counts"], str)
    assert np.array_equal(rle_to_mask(rle), mask)


def test_rle_roundtrip_empty_mask():
    mask = np.zeros((40, 60), dtype=bool)
    assert not rle_to_mask(mask_to_rle(mask)).any()


def test_xyxy_to_xywh():
    assert xyxy_to_xywh(np.array([[10, 20, 50, 80]])).tolist() == [[10.0, 20.0, 40.0, 60.0]]
    assert xyxy_to_xywh(np.zeros((0, 4))).shape == (0, 4)


def test_prediction_to_coco():
    mask = _ring_mask()
    coco = prediction_to_coco(60, 40, np.array([[10, 5, 50, 30]]), mask[None], np.array([0.9]), file_name="a.jpg")
    restored = json.loads(json.dumps(coco))

    assert restored["image"] == {"file_name": "a.jpg", "width": 60, "height": 40}
    assert restored["categories"] == [{"id": 1, "name": "grape_cluster"}]
    (annotation,) = restored["annotations"]
    assert annotation["bbox"] == [10.0, 5.0, 40.0, 25.0]
    assert annotation["area"] == int(mask.sum())
    assert annotation["score"] == 0.9
    assert np.array_equal(rle_to_mask(annotation["segmentation"]), mask)


def test_prediction_to_coco_without_detections():
    coco = prediction_to_coco(60, 40, np.zeros((0, 4)), np.zeros((0, 40, 60), dtype=bool), np.zeros(0))
    assert coco["annotations"] == []
