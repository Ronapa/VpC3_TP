"""Helpers de visualización: cajas de Florence-2 y máscaras de SAM."""
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_florence_boxes(image: Image.Image, parsed: dict, task_key: str, title: str | None = None) -> None:
    """Dibuja las cajas que devuelve Florence-2 sobre la imagen.

    Soporta tanto el formato de `<OD>` (`{"bboxes": [...], "labels": [...]}`) como
    el de `<OPEN_VOCABULARY_DETECTION>` (`{"bboxes": [...], "bboxes_labels": [...]}`).
    """
    result = parsed[task_key]
    labels_key = "labels" if "labels" in result else "bboxes_labels"
    fig, ax = plt.subplots(figsize=(8, 8 * image.size[1] / image.size[0]))
    ax.imshow(image)
    for box, label in zip(result["bboxes"], result[labels_key]):
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="lime", facecolor="none")
        )
        ax.text(x1, max(y1 - 5, 0), label, color="black", fontsize=9, backgroundcolor="lime")
    ax.set_title(title or f"{task_key} - {len(result['bboxes'])} detecciones")
    ax.axis("off")
    plt.show()


def plot_sam_mask(
    image_np: np.ndarray,
    mask: np.ndarray,
    box: np.ndarray | None = None,
    title: str | None = None,
    alpha: float = 0.5,
) -> None:
    """Overlay de una máscara de SAM (modo promptable) sobre la imagen, con su caja de referencia."""
    overlay = image_np.copy()
    color = np.array([30, 255, 30], dtype=np.uint8)
    mask_2d = mask.squeeze().astype(bool)
    overlay[mask_2d] = (overlay[mask_2d] * (1 - alpha) + color * alpha).astype(np.uint8)

    fig, ax = plt.subplots(figsize=(8, 8 * image_np.shape[0] / image_np.shape[1]))
    ax.imshow(overlay)
    if box is not None:
        x1, y1, x2, y2 = box
        ax.add_patch(
            patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="red", facecolor="none")
        )
    ax.set_title(title or f"SAM promptable - mascara {mask_2d.sum()} px")
    ax.axis("off")
    plt.show()


def paint_masks(
    image_np: np.ndarray,
    masks,
    color: tuple[int, int, int] = (0, 90, 255),
    alpha: float = 0.55,
    edge_width: int = 4,
) -> np.ndarray:
    """Devuelve la imagen con las máscaras pintadas en un solo color y el borde de cada una en blanco,
    para distinguir instancias. `masks` es un iterable de máscaras booleanas 2D."""
    union = np.zeros(image_np.shape[:2], dtype=bool)
    edges = np.zeros(image_np.shape[:2], dtype=bool)
    for m in masks:
        interior = m.copy()
        interior[edge_width:, :] &= m[:-edge_width, :]
        interior[:-edge_width, :] &= m[edge_width:, :]
        interior[:, edge_width:] &= m[:, :-edge_width]
        interior[:, :-edge_width] &= m[:, edge_width:]
        union |= m
        edges |= m & ~interior
    overlay = image_np.copy()
    overlay[union] = (overlay[union] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
    overlay[edges] = 255
    return overlay
