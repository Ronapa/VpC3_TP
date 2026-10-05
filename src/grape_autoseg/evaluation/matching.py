"""Emparejamiento de cajas/máscaras predichas contra ground truth, vía asignación húngara sobre IoU.

Húngaro (no greedy) porque maximiza el IoU total emparejado considerando todas las
combinaciones a la vez; con cajas de racimos que se superponen, un greedy puede
quedarse con una asignación localmente buena pero globalmente subóptima.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment


def box_iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """IoU entre cada par de cajas (formato xyxy). Shape (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    boxes_a = np.asarray(boxes_a)
    boxes_b = np.asarray(boxes_b)

    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])

    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter

    return np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """IoU entre dos máscaras booleanas 2D."""
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


def match_boxes(
    pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float = 0.5
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Empareja predicciones con GT vía asignación húngara sobre la matriz de IoU.

    Devuelve (matches, unmatched_pred_idx, unmatched_gt_idx), donde `matches` es
    una lista de (idx_pred, idx_gt) con IoU >= iou_threshold.
    """
    iou = box_iou_matrix(pred_boxes, gt_boxes)
    if iou.size == 0:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

    row_idx, col_idx = linear_sum_assignment(-iou)  # maximizar IoU == minimizar -IoU

    matches = [(int(r), int(c)) for r, c in zip(row_idx, col_idx) if iou[r, c] >= iou_threshold]
    matched_pred = {m[0] for m in matches}
    matched_gt = {m[1] for m in matches}
    unmatched_pred = [i for i in range(len(pred_boxes)) if i not in matched_pred]
    unmatched_gt = [i for i in range(len(gt_boxes)) if i not in matched_gt]

    return matches, unmatched_pred, unmatched_gt
