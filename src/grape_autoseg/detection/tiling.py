"""Detección por tiles: partir la imagen en una grilla con solape, detectar en cada tile y fusionar.

Motivación: Florence-2 redimensiona cualquier entrada a 768x768, así que en una
imagen de 2048 px de ancho un racimo mediano queda en ~63 px. Detectar por tile
sube la resolución efectiva y baja la cantidad de racimos que el modelo tiene
que enumerar en cada pasada.
"""
from collections.abc import Callable

import numpy as np
from PIL import Image

from grape_autoseg.evaluation.matching import box_iou_matrix


def make_tiles(width: int, height: int, cols: int, rows: int, overlap: float = 0.2) -> np.ndarray:
    """Grilla de `cols` x `rows` tiles (xyxy, en píxeles) que cubre toda la imagen.

    `overlap` es la fracción del lado del tile que se comparte con el vecino.
    """

    def _starts_and_size(total: int, n: int) -> tuple[np.ndarray, int]:
        size = int(np.ceil(total / (n - (n - 1) * overlap)))
        starts = np.linspace(0, total - size, n).round().astype(int)
        return starts, size

    xs, tile_w = _starts_and_size(width, cols)
    ys, tile_h = _starts_and_size(height, rows)
    return np.array([[x, y, x + tile_w, y + tile_h] for y in ys for x in xs])


def detect_tiled(
    image: Image.Image,
    detect_fn: Callable[[Image.Image], np.ndarray],
    cols: int,
    rows: int,
    overlap: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Corre `detect_fn` sobre cada tile y lleva las cajas a coordenadas de la imagen completa.

    Devuelve (boxes, tile_idx, tiles): las cajas crudas sin fusionar, el índice
    del tile del que salió cada una y la grilla de tiles usada.
    """
    tiles = make_tiles(*image.size, cols, rows, overlap)
    boxes, tile_idx = [], []
    for k, (x1, y1, x2, y2) in enumerate(tiles):
        tile_boxes = detect_fn(image.crop((x1, y1, x2, y2)))
        if len(tile_boxes) == 0:
            continue
        boxes.append(tile_boxes + np.array([x1, y1, x1, y1]))
        tile_idx.append(np.full(len(tile_boxes), k))

    if not boxes:
        return np.zeros((0, 4)), np.zeros(0, dtype=int), tiles
    return np.concatenate(boxes), np.concatenate(tile_idx), tiles


def box_areas(boxes: np.ndarray) -> np.ndarray:
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def merge_boxes(boxes: np.ndarray, iou_threshold: float = 0.5) -> np.ndarray:
    """Fusiona duplicados entre tiles. Devuelve los índices de las cajas que quedan.

    Florence-2 no da score, así que la prioridad es por área: gana la caja más
    grande, porque entre dos detecciones del mismo racimo la más chica suele ser
    la que quedó cortada por el borde de un tile.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)

    iou = box_iou_matrix(boxes, boxes)

    keep: list[int] = []
    for i in np.argsort(-box_areas(boxes)):
        if keep and iou[i, keep].max() >= iou_threshold:
            continue
        keep.append(int(i))
    return np.array(sorted(keep))


def drop_group_boxes(boxes: np.ndarray, containment_threshold: float = 0.7, min_contained: int = 2) -> np.ndarray:
    """Descarta las cajas que agrupan varios racimos. Devuelve los índices de las que quedan.

    Una caja se considera de grupo si contiene `min_contained` o más de las otras
    cajas (cada una con al menos `containment_threshold` de su área adentro).
    Pensada para la pasada de imagen completa: sobre tiles, un racimo con varias
    bayas detectadas adentro también cumpliría la condición.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)

    areas = box_areas(boxes)
    iou = box_iou_matrix(boxes, boxes)
    inter = iou * (areas[:, None] + areas[None, :]) / (1 + iou)
    inside = inter / np.maximum(areas[None, :], 1e-9)  # [i, j] = fracción de j dentro de i
    np.fill_diagonal(inside, 0)
    return np.flatnonzero((inside >= containment_threshold).sum(axis=1) < min_contained)


def merge_masks(
    masks: np.ndarray,
    scores: np.ndarray,
    iou_threshold: float = 0.5,
    containment_threshold: float | None = None,
    group_min_fraction: float | None = None,
) -> np.ndarray:
    """Deduplica máscaras (N, H, W) booleanas. Devuelve los índices de las que quedan.

    NMS por IoU de máscara con prioridad por `scores` (el score de calidad que
    da SAM): entre dos máscaras del mismo racimo queda la de mayor score.

    Con `containment_threshold` se resuelven después las máscaras anidadas: se
    descarta toda máscara que tenga esa fracción de su área dentro de otra más
    grande que haya quedado (baya o pedazo de racimo dentro del racimo entero).
    Sola, esa regla también descarta un racimo real si quedó dentro de una
    máscara que agrupa varios.

    Con `group_min_fraction` se distingue ese caso antes de aplicarla: una
    máscara que contiene dos o más máscaras, cada una de al menos esa fracción
    de su área, se considera un grupo de racimos y se descarta ella, no las de
    adentro. Las bayas no cuentan: son una fracción mucho más chica del racimo.
    """
    if len(masks) == 0:
        return np.zeros(0, dtype=int)

    flat = masks.reshape(len(masks), -1).astype(np.float32)
    areas = flat.sum(axis=1)
    inter = flat @ flat.T
    union = areas[:, None] + areas[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)

    keep: list[int] = []
    for i in np.argsort(-scores):
        if keep and iou[i, keep].max() >= iou_threshold:
            continue
        keep.append(int(i))

    if containment_threshold is not None:
        inside = inter / np.maximum(areas[:, None], 1)  # [i, j] = fracción de i dentro de j

        def contained_in(i: int, j: int) -> bool:
            return areas[j] > areas[i] and inside[i, j] >= containment_threshold

        if group_min_fraction is not None:
            keep = [
                j
                for j in keep
                if sum(contained_in(i, j) and areas[i] >= group_min_fraction * areas[j] for i in keep) < 2
            ]
        keep = [i for i in keep if not any(contained_in(i, j) for j in keep)]
    return np.array(sorted(keep), dtype=int)
