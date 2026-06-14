# Aerial Object Detection — Drones, Planes, Helicopters

Object detection of aerial objects (3 classes: **AirPlane**, **Drone**, **Helicopter**)
with YOLO-family models, built on [PyTorch Lightning](https://lightning.ai/),
[Fiddle](https://github.com/google/fiddle), and [Weights & Biases](https://wandb.ai/).

The project compares a **from-scratch YOLOv1-style baseline** against **fine-tuned
Ultralytics models** (YOLOv8n, YOLO26n), and runs ablations to explain *why* the
modern models win — separating the effect of input resolution, pretrained weights,
and architecture.

---

## Problem & dataset

[Drone Detection](https://www.kaggle.com/datasets/cybersimar08/drone-detection) (Kaggle,
Roboflow YOLO export), 3 classes: `AirPlane=0`, `Drone=1`, `Helicopter=2`.

| split | images | objects | empty (background-only) |
|-------|-------:|--------:|------------------------:|
| train | 10,799 | 8,999   | 1,979 (18.3%)           |
| valid |    603 |   497   |   110 (18.2%)           |
| test  |    596 |   479   |   123 (20.6%)           |

Notable properties (see `notebooks/dataset_statistics.ipynb`): class imbalance
(Drone > Helicopter ≈ AirPlane), ~18–20% background-only images, and **small objects**
(median box side ≈ 7px at 128px input, ≈ 37px at 640px) — which directly motivates the
resolution ablation.

---

## Models

| config | model | weights | input | notes |
|--------|-------|---------|------:|-------|
| `src/config/yolov1.py` | `YoloV1Baseline` | from scratch | 128 | grid head, custom YOLOv1-style loss |
| `src/config/yolov1_256.py` | `YoloV1Baseline` | from scratch | 256 | **resolution ablation** |
| `src/config/yolov8.py` | `UltralyticsDetectionModel` (yolov8n) | COCO-pretrained | 640 | transfer learning |
| `src/config/yolov8_scratch.py` | `UltralyticsDetectionModel` (yolov8n) | random | 640 | **pretraining ablation** |
| `src/config/yolo26.py` | `UltralyticsDetectionModel` (yolo26n) | COCO-pretrained | 640 | transfer learning |

**Ablation logic** (notebooks/model_comparison.ipynb produces the table):

| comparison | isolates |
|------------|----------|
| yolov1 (128) vs yolov1 (256) | input resolution |
| yolov8 scratch vs yolov8 | pretrained weights (architecture fixed) |
| yolov1 (256) vs yolov8 scratch | architecture (resolution & no-pretraining controlled) |

All models are evaluated on the **same test split** with `torchmetrics`
`MeanAveragePrecision` (mAP50, mAP50-95, and per-class AP).

---

## Installation

Targets **Python 3.12**.

### uv (recommended)

```bash
uv python install 3.12   # only if you don't already have CPython 3.12
uv sync
```

`uv sync` creates `.venv`, resolves dependencies, and installs this package in editable
mode. Prefix commands with `uv run` (e.g. `uv run python scripts/train_model.py …`) or
activate the venv: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).

### PyTorch and CUDA

`pyproject.toml` cannot pick a PyTorch wheel for your GPU/CUDA. For a specific build,
install PyTorch **before** `uv sync`:

```bash
uv venv --python 3.12
source .venv/bin/activate            # Windows: .venv\Scripts\activate
# CPU-only (macOS installs CPU by default):
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.6:
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
uv sync
```

Other CUDA versions: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Dataset setup

`data/` is gitignored. Download via the [Kaggle API](https://github.com/Kaggle/kaggle-api):

```bash
kaggle datasets download -d cybersimar08/drone-detection -p data --unzip
```

Arrange the splits directly under `data/` (the Kaggle export may nest them in a subfolder):

```text
data/
├── train/   (images/  labels/)
├── valid/
└── test/
```

Each `labels/*.txt` holds YOLO lines `class_id x_center y_center width height` normalized
to `[0,1]`. `DroneDetectionYoloDataModule` resizes to a square, scales to `[0,1]`, and
re-normalizes boxes to that square. Batches are
`{"images": (B,3,H,W), "targets": List[(n_i,5)]}` with target columns
`cls, xc, yc, w, h`. If your export uses `val/`, pass `val_dir="val"`.

---

## Running experiments

One command per config; `scripts/train_model.py` is model-agnostic. After `fit`, it
evaluates the best checkpoint (by `val/mAP50`) on the test split and logs
`test/mAP50`, `test/mAP50-95`, and per-class `test/AP_*`.

```bash
python scripts/train_model.py src/config/yolov1.py
python scripts/train_model.py src/config/yolov1_256.py
python scripts/train_model.py src/config/yolov8.py
python scripts/train_model.py src/config/yolov8_scratch.py
python scripts/train_model.py src/config/yolo26.py
```

Disable W&B with `--no_wandb`. Resume an interrupted run with
`python scripts/train_model.py --resume_run_name <run_name>`. Checkpoints are written to
`logs/<run_name>/`.

---

## Reproducing the analysis

Two notebooks (run after training, from the repo root or `notebooks/`):

- **`notebooks/dataset_statistics.ipynb`** — class distribution, background-only fraction,
  object-size and objects-per-image distributions, sample images with GT boxes. Needs only
  the dataset (runs in seconds).
- **`notebooks/model_comparison.ipynb`** — loads the best local checkpoint of each model,
  runs them on the test split, and produces: the comparison table (mAP + per-class AP),
  side-by-side bounding-box visualizations, per-model confusion matrices, and the ablation
  table. Models without a local checkpoint are skipped automatically.

---

## Project structure

```text
.
├── data/                              # dataset (gitignored): train/ valid/ test/
├── scripts/
│   └── train_model.py                 # single entry point (fit + test) for all configs
├── notebooks/
│   ├── dataset_statistics.ipynb       # dataset EDA
│   └── model_comparison.ipynb         # cross-model evaluation + visualizations
└── src/
    ├── config/                        # Fiddle configs (one per experiment)
    │   ├── schemas.py  constants.py
    │   ├── yolov1.py  yolov1_256.py
    │   ├── yolov8.py  yolov8_scratch.py  yolo26.py
    ├── datasets/
    │   └── drone_detection_yolo.py    # DroneDetectionYoloDataModule (YOLO-folder data)
    ├── models/
    │   ├── yolov1_baseline.py         # from-scratch YOLOv1-style LightningModule
    │   ├── ultralytics_yolo.py        # Ultralytics YOLO wrapper (v8 / 26, pretrained flag)
    │   ├── architectures/grid_convolutional.py
    │   └── modules/conv_block.py
    └── utils/
        ├── config.py                  # load / resume Fiddle configs
        ├── evaluation.py              # checkpoint loading + test-pass helpers
        └── visualization.py           # box drawing + confusion-matrix plotting
```

---

## Key abstractions

- **`LightningModule`** (`src/models/`) — architecture, loss, optimizer, train/val/test
  steps. New model = new module; the training script doesn't change.
- **`LightningDataModule`** (`src/datasets/`) — dataset, splits, transforms, dataloaders.
- **Fiddle configs** (`src/config/`) — each returns a `fdl.Config[ExperimentConfig]`.
  Every object must be wrapped in `fdl.Config`/`fdl.Partial` (not instantiated directly),
  so Fiddle can serialize the fully-resolved config for reproducibility. Configs can call
  another config's `build_config()` and override fields (e.g. the ablation configs mirror
  their baselines with a few changes).
```
