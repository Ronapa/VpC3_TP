import numpy as np

from grape_autoseg.detection.tiling import drop_group_boxes, merge_masks


def _mask(rows: slice, cols: slice) -> np.ndarray:
    mask = np.zeros((20, 20), dtype=bool)
    mask[rows, cols] = True
    return mask


BIG = _mask(slice(0, 10), slice(0, 20))  # 200 px
LEFT = _mask(slice(0, 10), slice(0, 5))  # 50 px, dentro de BIG
RIGHT = _mask(slice(0, 10), slice(10, 15))  # 50 px, dentro de BIG


def test_drop_group_boxes_removes_box_containing_two_others():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 40, 40], [50, 50, 90, 90]])
    assert drop_group_boxes(boxes).tolist() == [1, 2]


def test_drop_group_boxes_keeps_box_containing_only_one():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 40, 40]])
    assert drop_group_boxes(boxes).tolist() == [0, 1]


def test_drop_group_boxes_ignores_partial_overlap():
    # las dos cajas chicas tienen menos del 70 % de su área dentro de la grande
    boxes = np.array([[0, 0, 100, 100], [80, 0, 180, 40], [0, 90, 50, 190]])
    assert drop_group_boxes(boxes).tolist() == [0, 1, 2]


def test_drop_group_boxes_min_contained():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 40, 40], [50, 50, 90, 90]])
    assert drop_group_boxes(boxes, min_contained=3).tolist() == [0, 1, 2]


def test_drop_group_boxes_empty():
    assert drop_group_boxes(np.zeros((0, 4))).tolist() == []


def test_merge_masks_keeps_higher_score_of_duplicates():
    masks = np.stack([_mask(slice(0, 10), slice(0, 10)), _mask(slice(0, 10), slice(0, 8))])  # IoU 0.8
    assert merge_masks(masks, np.array([0.5, 0.9])).tolist() == [1]
    assert merge_masks(masks, np.array([0.9, 0.5])).tolist() == [0]


def test_merge_masks_keeps_disjoint_masks():
    masks = np.stack([LEFT, RIGHT])
    assert merge_masks(masks, np.array([0.9, 0.8])).tolist() == [0, 1]


def test_merge_masks_keeps_nested_masks_by_default():
    # IoU 0.25: el NMS solo no las considera duplicados
    masks = np.stack([BIG, LEFT])
    assert merge_masks(masks, np.array([0.9, 0.8])).tolist() == [0, 1]


def test_merge_masks_containment_drops_nested_masks():
    masks = np.stack([BIG, LEFT, RIGHT])
    scores = np.array([0.9, 0.8, 0.7])
    assert merge_masks(masks, scores, containment_threshold=0.7).tolist() == [0]


def test_merge_masks_group_drops_outer_mask_instead():
    # LEFT y RIGHT ocupan cada una el 25 % de BIG: BIG es un grupo de racimos
    masks = np.stack([BIG, LEFT, RIGHT])
    scores = np.array([0.9, 0.8, 0.7])
    assert merge_masks(masks, scores, containment_threshold=0.7, group_min_fraction=0.2).tolist() == [1, 2]


def test_merge_masks_group_ignores_small_inner_masks():
    # dos "bayas" del 2 % del área no convierten a BIG en un grupo
    berries = [_mask(slice(0, 2), slice(0, 2)), _mask(slice(5, 7), slice(10, 12))]
    masks = np.stack([BIG, *berries])
    scores = np.array([0.9, 0.8, 0.7])
    assert merge_masks(masks, scores, containment_threshold=0.7, group_min_fraction=0.2).tolist() == [0]


def test_merge_masks_empty():
    assert merge_masks(np.zeros((0, 20, 20), dtype=bool), np.zeros(0)).tolist() == []
