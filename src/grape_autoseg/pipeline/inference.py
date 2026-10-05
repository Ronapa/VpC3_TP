"""Pipeline final de inferencia: Florence-2 detecta los racimos a partir de una frase y SAM segmenta cada caja."""
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoProcessor, Florence2ForConditionalGeneration

from grape_autoseg.detection.tiling import drop_group_boxes, merge_masks
from grape_autoseg.pipeline.coco import prediction_to_coco

CHECKPOINTS_DIR = Path(__file__).resolve().parents[3] / "data" / "checkpoints"
SAM_CHECKPOINTS = {"vit_b": "sam_vit_b_01ec64.pth", "vit_h": "sam_vit_h_4b8939.pth"}
TASK_PROMPT = "<OPEN_VOCABULARY_DETECTION>"


@dataclass
class PipelineConfig:
    florence_size: str = "base"
    phrase: str = "grapes"
    num_beams: int = 3
    # cada caja son 4 tokens: con un límite bajo la lista de cajas se corta
    max_new_tokens: int = 256
    sam_model: str = "vit_h"
    sam_checkpoint: str | None = None  # None: el de `sam_model` en data/checkpoints/
    drop_group_boxes: bool = True
    mask_iou_threshold: float = 0.5
    # el NMS de máscara se calcula submuestreado, como en la evaluación; las máscaras devueltas son a resolución completa
    mask_stride: int = 8
    device: str | None = None  # None: cuda si está disponible


@dataclass
class Prediction:
    width: int
    height: int
    boxes_xyxy: np.ndarray  # (N, 4), caja ajustada a cada máscara
    masks: np.ndarray  # (N, H, W) booleanas
    scores: np.ndarray  # (N,), calidad de máscara que estima SAM (Florence-2 no da confianza de detección)
    florence_boxes: np.ndarray  # (M, 4), todas las cajas de Florence-2
    group_box_idx: np.ndarray  # índices de `florence_boxes` descartados por ser cajas de grupo
    config: dict

    def to_coco(self, file_name: str | None = None) -> dict:
        return prediction_to_coco(self.width, self.height, self.boxes_xyxy, self.masks, self.scores, file_name, self.config)


class GrapePipeline:
    """Carga Florence-2 y SAM una vez; `predict` se puede llamar con tantas imágenes como haga falta."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")

        model_id = f"florence-community/Florence-2-{self.config.florence_size}"
        self.florence_model = Florence2ForConditionalGeneration.from_pretrained(model_id, device_map=self.device).eval()
        self.florence_processor = AutoProcessor.from_pretrained(model_id)

        checkpoint = self.config.sam_checkpoint or CHECKPOINTS_DIR / SAM_CHECKPOINTS[self.config.sam_model]
        sam = sam_model_registry[self.config.sam_model](checkpoint=str(checkpoint)).to(self.device)
        self.sam_predictor = SamPredictor(sam)

    def detect(self, image: Image.Image) -> np.ndarray:
        """Cajas xyxy de Florence-2 para la frase configurada."""
        inputs = self.florence_processor(text=TASK_PROMPT + self.config.phrase, images=image, return_tensors="pt").to(
            self.device, torch.float32
        )
        with torch.no_grad():
            ids = self.florence_model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens, num_beams=self.config.num_beams
            )
        text = self.florence_processor.batch_decode(ids, skip_special_tokens=False)[0]
        parsed = self.florence_processor.post_process_generation(text, task=TASK_PROMPT, image_size=image.size)
        boxes = parsed[TASK_PROMPT]["bboxes"]
        return np.array(boxes, dtype=float).reshape(-1, 4) if boxes else np.zeros((0, 4))

    def predict(self, image: Image.Image) -> Prediction:
        image = image.convert("RGB")
        image_np = np.array(image)

        florence_boxes = self.detect(image)
        keep = drop_group_boxes(florence_boxes) if self.config.drop_group_boxes else np.arange(len(florence_boxes))
        masks, scores = self._segment(image_np, florence_boxes[keep])

        return Prediction(
            width=image.width,
            height=image.height,
            boxes_xyxy=_mask_boxes(masks),
            masks=masks,
            scores=scores,
            florence_boxes=florence_boxes,
            group_box_idx=np.setdiff1d(np.arange(len(florence_boxes)), keep),
            config=asdict(self.config),
        )

    def _segment(self, image_np: np.ndarray, boxes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """SAM con prompt de caja y NMS de máscara, con prioridad por el score de SAM."""
        stride = self.config.mask_stride
        masks, scores = [], []
        if len(boxes):
            self.sam_predictor.set_image(image_np)
        for box in boxes:
            m, s, _ = self.sam_predictor.predict(box=np.array(box), multimask_output=False)
            if m[0][::stride, ::stride].any():
                masks.append(m[0])
                scores.append(float(s[0]))
        if not masks:
            return np.zeros((0, *image_np.shape[:2]), dtype=bool), np.zeros(0)

        masks, scores = np.stack(masks), np.array(scores)
        keep = merge_masks(masks[:, ::stride, ::stride], scores, iou_threshold=self.config.mask_iou_threshold)
        return masks[keep], scores[keep]


def _mask_boxes(masks: np.ndarray) -> np.ndarray:
    boxes = []
    for m in masks:
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return np.array(boxes, dtype=float).reshape(-1, 4)
