"""Métricas de detección para un detector sin score: un único punto de operación, sin curva precisión-recall."""
from dataclasses import dataclass

import numpy as np

from .matching import box_iou_matrix, match_boxes


@dataclass
class ImageDetectionResult:
    image_name: str
    n_pred: int
    n_gt: int
    n_matches: int
    matches: list[tuple[int, int]]
    unmatched_pred: list[int]
    unmatched_gt: list[int]
    match_ious: list[float]

    @property
    def precision(self) -> float:
        return self.n_matches / self.n_pred if self.n_pred > 0 else 0.0

    @property
    def recall(self) -> float:
        return self.n_matches / self.n_gt if self.n_gt > 0 else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def count_error(self) -> int:
        """Error de conteo: predichas - reales (positivo = sobre-detección)."""
        return self.n_pred - self.n_gt


def evaluate_detection(
    image_name: str, pred_boxes: np.ndarray, gt_boxes: np.ndarray, iou_threshold: float = 0.5
) -> ImageDetectionResult:
    matches, unmatched_pred, unmatched_gt = match_boxes(pred_boxes, gt_boxes, iou_threshold)
    iou = box_iou_matrix(pred_boxes, gt_boxes)
    match_ious = [float(iou[r, c]) for r, c in matches]
    return ImageDetectionResult(
        image_name=image_name,
        n_pred=len(pred_boxes),
        n_gt=len(gt_boxes),
        n_matches=len(matches),
        matches=matches,
        unmatched_pred=unmatched_pred,
        unmatched_gt=unmatched_gt,
        match_ious=match_ious,
    )


def aggregate_detection_results(results: list[ImageDetectionResult]) -> dict:
    """Agrega micro (sobre el total de cajas) y macro (promedio por imagen)."""
    total_pred = sum(r.n_pred for r in results)
    total_gt = sum(r.n_gt for r in results)
    total_matches = sum(r.n_matches for r in results)

    micro_precision = total_matches / total_pred if total_pred > 0 else 0.0
    micro_recall = total_matches / total_gt if total_gt > 0 else 0.0
    micro_f1 = (
        2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        if (micro_precision + micro_recall) > 0
        else 0.0
    )

    macro_precision = float(np.mean([r.precision for r in results])) if results else 0.0
    macro_recall = float(np.mean([r.recall for r in results])) if results else 0.0
    macro_f1 = float(np.mean([r.f1 for r in results])) if results else 0.0

    count_errors = [r.count_error for r in results]
    mae_count = float(np.mean(np.abs(count_errors))) if count_errors else 0.0

    all_match_ious = [iou for r in results for iou in r.match_ious]

    return {
        "n_images": len(results),
        "total_pred": total_pred,
        "total_gt": total_gt,
        "total_matches": total_matches,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "micro_f1": micro_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "mae_count": mae_count,
        "mean_match_iou": float(np.mean(all_match_ious)) if all_match_ious else 0.0,
    }
