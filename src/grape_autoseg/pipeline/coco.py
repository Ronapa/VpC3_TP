"""Salida del pipeline en formato COCO: caja [x, y, ancho, alto] y máscara en RLE comprimido."""
import numpy as np
from pycocotools import mask as mask_utils

CATEGORY = {"id": 1, "name": "grape_cluster"}


def mask_to_rle(mask: np.ndarray) -> dict:
    """Máscara booleana (H, W) a RLE de COCO, con `counts` como texto para poder serializarlo a JSON."""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {"size": [int(s) for s in rle["size"]], "counts": rle["counts"].decode("ascii")}


def rle_to_mask(rle: dict) -> np.ndarray:
    return mask_utils.decode({"size": rle["size"], "counts": rle["counts"].encode("ascii")}).astype(bool)


def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    return np.column_stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]])


def prediction_to_coco(
    width: int,
    height: int,
    boxes_xyxy: np.ndarray,
    masks: np.ndarray,
    scores: np.ndarray,
    file_name: str | None = None,
    config: dict | None = None,
) -> dict:
    """Diccionario serializable a JSON con una anotación por racimo, en el formato de resultados de COCO."""
    annotations = [
        {
            "id": k + 1,
            "category_id": CATEGORY["id"],
            "bbox": [float(v) for v in bbox],
            "area": int(mask.sum()),
            "score": float(score),
            "segmentation": mask_to_rle(mask),
        }
        for k, (bbox, mask, score) in enumerate(zip(xyxy_to_xywh(boxes_xyxy), masks, scores))
    ]
    return {
        "image": {"file_name": file_name, "width": int(width), "height": int(height)},
        "categories": [CATEGORY],
        "annotations": annotations,
        "config": config or {},
    }
