"""Carga de WGISD: imágenes con sus cajas (en píxeles) y máscaras, cuando existen.

Fuente de verdad: archivos YOLO (`.txt`) y `.npz` originales del dataset, no las
anotaciones COCO de terceros.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

VARIETY_COLOR = {
    "CDY": "blanca",
    "SVB": "blanca",
    "CFR": "tinta",
    "CSV": "tinta",
    "SYH": "tinta",
}


@dataclass
class WgisdSample:
    name: str
    image: Image.Image
    boxes_xyxy: np.ndarray  # (N, 4) en píxeles
    masks: np.ndarray | None  # (H, W, N) o None si no hay máscara para esta imagen
    variety: str
    color: str


def _yolo_to_xyxy(boxes_yolo: np.ndarray, width: int, height: int) -> np.ndarray:
    cx, cy, w, h = boxes_yolo[:, 1], boxes_yolo[:, 2], boxes_yolo[:, 3], boxes_yolo[:, 4]
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return np.stack([x1, y1, x2, y2], axis=1)


def load_sample(images_dir: Path, name: str) -> WgisdSample:
    """Carga una imagen de WGISD por su nombre base (sin extensión)."""
    image = Image.open(images_dir / f"{name}.jpg").convert("RGB")
    width, height = image.size

    boxes_yolo = np.loadtxt(images_dir / f"{name}.txt").reshape(-1, 5)
    boxes_xyxy = _yolo_to_xyxy(boxes_yolo, width, height)

    mask_path = images_dir / f"{name}.npz"
    masks = None
    if mask_path.exists():
        npz = np.load(mask_path)
        masks = npz[list(npz.keys())[0]]  # (H, W, n_clusters), alineado por índice con boxes

    prefix = name[:3]
    return WgisdSample(
        name=name,
        image=image,
        boxes_xyxy=boxes_xyxy,
        masks=masks,
        variety=prefix,
        color=VARIETY_COLOR.get(prefix, "desconocido"),
    )


def list_masked_names(images_dir: Path) -> list[str]:
    """Nombres (sin extensión) de todas las imágenes que tienen máscara `.npz`."""
    return sorted(p.stem for p in images_dir.glob("*.npz"))
