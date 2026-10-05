# grape-autoseg

Detección y segmentación automática de racimos de uva en imágenes de viñedo,
usando modelos preentrenados (Florence-2 + SAM), sin entrenamiento propio.

Proyecto final individual — materia de posgrado sobre Transformers de visión,
modelos multimodales y Segment Anything.

## Estado actual

Los experimentos están en los notebooks (`notebooks/`). El
pipeline final (Florence-2 + SAM) está en `src/grape_autoseg/pipeline/` y se
puede correr sobre cualquier imagen: ver la sección "Inferencia".

## Dataset: WGISD

Embrapa Wine Grape Instance Segmentation Dataset.
Repo: https://github.com/thsant/wgisd — también en
[Hugging Face](https://huggingface.co/datasets/thsant/wgisd).

**Licencia:** CC BY-NC 4.0 — uso académico permitido.

**Cita:**
```
Santos, T.T.; de Souza, L.L.; dos Santos, A.A.; Avila, S. Grape detection,
segmentation and tracking using deep neural networks and three-dimensional
association. Computers and Electronics in Agriculture, v. 170, 2020.
```

## Instalación

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Cómo correr

```bash
# descargar el dataset
python scripts/download_wgisd.py --dest data/wgisd

# notebooks, en orden
jupyter notebook notebooks/
```

El notebook 04 necesita los checkpoints de SAM en `data/checkpoints/` (ver
[`docs/inferencia.md`](docs/inferencia.md)).

- `notebooks/01_dataset_loading.ipynb` — descarga WGISD y revisa su estructura
  (cajas, máscaras, split oficial).
- `notebooks/02_inspection.ipynb` — inspección del dataset: conteos por variedad,
  tamaños de racimo y overlays de las máscaras anotadas.
- `notebooks/04_florence_sam_prompts.ipynb` — los experimentos: frases, tamaño de
  Florence-2, beam search, tiling, cajas de grupo y tamaño de SAM, evaluados sobre
  las 137 imágenes con máscara. Cada prueba guarda sus métricas y la última celda
  las compara.

## Inferencia

El pipeline final detecta los racimos con Florence-2, descarta las cajas de
grupo, segmenta cada caja con SAM y deduplica con NMS de máscara.

```bash
python scripts/predict.py imagen1.jpg imagen2.jpg --overlay
```

Guarda en `outputs/predict/` un JSON por imagen con cajas y máscaras en formato
COCO y, con `--overlay`, la imagen con las máscaras dibujadas.

Guía completa (checkpoints, opciones, uso desde Python, formato de salida y cómo
leer los resultados): [`docs/inferencia.md`](docs/inferencia.md).

## Tests

```bash
pytest tests/
```
