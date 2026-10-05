"""Métricas de segmentación: IoU de máscara por racimo emparejado."""
from dataclasses import dataclass

import numpy as np

from .matching import mask_iou


@dataclass
class MaskMatchResult:
    image_name: str
    mask_ious: list[float]  # uno por par (pred, gt) ya emparejado a nivel caja

    @property
    def mean_iou(self) -> float:
        return float(np.mean(self.mask_ious)) if self.mask_ious else 0.0

    @property
    def pct_above_75(self) -> float:
        if not self.mask_ious:
            return 0.0
        return float(np.mean([iou >= 0.75 for iou in self.mask_ious]))


def evaluate_masks_for_matches(
    image_name: str,
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    matches: list[tuple[int, int]],
) -> MaskMatchResult:
    """IoU de máscara para cada par (idx_pred, idx_gt) que ya hizo match a nivel caja.

    No vuelve a buscar la mejor máscara GT: usa la que corresponde a la caja con la
    que esa predicción fue emparejada (ver notebook 01 — alineación por índice).
    """
    ious = [mask_iou(pred_masks[p_idx], gt_masks[g_idx]) for p_idx, g_idx in matches]
    return MaskMatchResult(image_name=image_name, mask_ious=ious)


def aggregate_mask_results(results: list[MaskMatchResult]) -> dict:
    all_ious = [iou for r in results for iou in r.mask_ious]
    return {
        "n_images": len(results),
        "n_matched_masks": len(all_ious),
        "mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "pct_above_75": float(np.mean([iou >= 0.75 for iou in all_ious])) if all_ious else 0.0,
    }
