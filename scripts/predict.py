"""Corre el pipeline final (Florence-2 + SAM) sobre una o más imágenes.

Por cada imagen guarda `<nombre>.json` con cajas y máscaras en formato COCO
(bbox [x, y, ancho, alto] y máscara RLE) y, con `--overlay`, `<nombre>_overlay.png`.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from PIL import Image, ImageDraw

from grape_autoseg.pipeline.inference import SAM_CHECKPOINTS, GrapePipeline, PipelineConfig
from grape_autoseg.viz.overlays import paint_masks


def save_overlay(path: Path, image: Image.Image, prediction) -> None:
    overlay = Image.fromarray(paint_masks(np.array(image), prediction.masks))
    draw = ImageDraw.Draw(overlay)
    for box in prediction.boxes_xyxy:
        draw.rectangle([float(v) for v in box], outline="yellow", width=3)
    overlay.save(path)


if __name__ == "__main__":
    defaults = PipelineConfig()
    parser = argparse.ArgumentParser(description="Detección y segmentación de racimos con Florence-2 + SAM")
    parser.add_argument("images", nargs="+", type=Path, help="imágenes de entrada")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/predict"), help="carpeta de salida")
    parser.add_argument("--overlay", action="store_true", help="guardar también la imagen con máscaras y cajas")
    parser.add_argument("--sam-model", choices=sorted(SAM_CHECKPOINTS), default=defaults.sam_model)
    parser.add_argument("--florence-size", choices=["base", "large"], default=defaults.florence_size)
    parser.add_argument("--phrase", default=defaults.phrase, help="frase para Florence-2")
    parser.add_argument("--num-beams", type=int, default=defaults.num_beams)
    parser.add_argument("--keep-group-boxes", action="store_true", help="no descartar las cajas que contienen a otras")
    args = parser.parse_args()

    pipeline = GrapePipeline(
        PipelineConfig(
            florence_size=args.florence_size,
            phrase=args.phrase,
            num_beams=args.num_beams,
            sam_model=args.sam_model,
            drop_group_boxes=not args.keep_group_boxes,
        )
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.images:
        image = Image.open(path).convert("RGB")
        prediction = pipeline.predict(image)

        out_path = args.out_dir / f"{path.stem}.json"
        out_path.write_text(json.dumps(prediction.to_coco(file_name=path.name)))
        if args.overlay:
            save_overlay(args.out_dir / f"{path.stem}_overlay.png", image, prediction)

        print(
            f"{path.name}: {len(prediction.florence_boxes)} cajas de Florence-2, "
            f"{len(prediction.group_box_idx)} de grupo descartadas, {len(prediction.masks)} mascaras -> {out_path}"
        )
