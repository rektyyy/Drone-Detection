# Neural Networks. Theory and Practice. 2026 Project Template

A template for training neural networks with [PyTorch Lightning](https://lightning.ai/), [Fiddle](https://github.com/google/fiddle), and [Weights & Biases](https://wandb.ai/).

The purpose of this template is to introduce PyTorch Lightning, Fiddle, and Weights & Biases together and showcase how they interact — not to provide a comprehensive guide to any of them individually. Each tool has far more functionality than is used here; the goal is to give you a working starting point that demonstrates a clean way to combine them.

The models included are not intended to be state-of-the-art. They exist solely to illustrate the abstractions that Lightning provides — specifically how a `LightningModule` encapsulates training logic and a `LightningDataModule` encapsulates data preparation, so that `train_model.py` stays model-agnostic and can train any experiment without modification.

---

## Installation

The project targets **Python 3.12** (`requires-python` in `pyproject.toml`).

### uv (recommended)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then from the repository root:

```bash
uv python install 3.12   # only needed if you do not already have CPython 3.12 on your PATH
uv sync
```

`uv sync` creates `.venv`, resolves dependencies from `pyproject.toml`, writes `uv.lock`, and installs this package in **editable** mode — Python loads `src/` directly, so code changes apply without reinstalling.

To use the environment in a shell: `source .venv/bin/activate` (Linux and macOS) or `.venv\Scripts\activate` (Windows). You can also prefix commands with `uv run` (for example `uv run python scripts/train_model.py …`).

### PyTorch and CUDA

`pyproject.toml` cannot tell whether you have a GPU or which CUDA version you need, so it cannot pick a PyTorch wheel for you. If you need a **specific** PyTorch build (CPU-only or a given CUDA release), install PyTorch **before** `uv sync`; the later sync will leave those packages in place if versions already satisfy the requirement:

```bash
uv venv --python 3.12
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# CPU-only (Linux/Windows — macOS installs CPU by default)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.6
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

uv sync
```

Index URLs for other CUDA versions: [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

### Alternative: micromamba

```bash
micromamba create -n nn_2026_template python=3.12
micromamba activate nn_2026_template
pip install -e .
```

```bash
micromamba install -c conda-forge libcst
```

---

## Project structure

```
.
├── data/                        # Datasets (see note below)
│   ├── utkface/
│   └── drone-detection/         # Kaggle drone dataset (YOLO layout; see below)
│
├── scripts/
│   └── train_model.py          # Single entry point for all experiments
│
├── notebooks/
│   └── train_classifier.ipynb  # Interactive equivalent of train_model.py
│
└── src/
    ├── config/
    │   ├── schemas.py           # ExperimentConfig and TrainingConfig dataclasses
    │   ├── constants.py         # W&B entity and project name
    │   ├── convnet_utkface_classifier.py
    │   ├── convnet_utkface_regressor.py
    │   ├── mlp_utkface_classifier.py
    │   └── lr_comparison/       # Example of Fiddle config inheritance (learning rate sweep)
    │       ├── 1e-2.py
    │       └── 1e-4.py
    │
    ├── datasets/
    │   ├── utkface.py                 # LightningDataModule for UTKFace
    │   └── drone_detection_yolo.py    # LightningDataModule for YOLO-folder drone data
    │
    ├── models/
    │   ├── classification_model.py   # LightningModule for classification
    │   ├── regression_model.py       # LightningModule for regression
    │   ├── architectures/
    │   │   ├── convolutional.py      # CNN backbone
    │   │   └── mlp.py               # MLP backbone
    │   └── modules/
    │       └── conv_block.py         # Reusable convolutional block
    │
    └── utils/
        └── config.py            # Helpers for loading and resuming Fiddle configs
```

### A note on `data/`

Large datasets are often stored on a separate drive rather than inside the project directory. Rather than hardcoding absolute paths in your configs, the recommended approach is to symlink the data directory into the project root:

```bash
# Linux / macOS — symlink the entire data directory at once
ln -s /path/to/your/data data

# Linux / macOS — or symlink individual datasets separately
ln -s /path/to/your/data/utkface data/utkface
ln -s /path/to/your/data/drone-detection data/drone-detection

# Windows (run as Administrator) — entire directory
mklink /D data C:\path\to\your\data

# Windows (run as Administrator) — individual datasets
mklink /D data\utkface C:\path\to\your\data\utkface
mklink /D data\drone-detection C:\path\to\your\data\drone-detection
```

This keeps all paths in the codebase relative (e.g. `data/utkface`), so configs and scripts work the same way on every machine regardless of where the data is physically stored.

### Drone detection dataset (Kaggle)

Download the **[Drone Detection](https://www.kaggle.com/datasets/cybersimar08/drone-detection)** dataset from Kaggle and place it under **`./data`** (the project gitignores `data/`, so nothing large is committed).

1. Install the [Kaggle API](https://github.com/Kaggle/kaggle-api) and add your API credentials (`~/.kaggle/kaggle.json`).
2. From the repository root:

```bash
kaggle datasets download -d cybersimar08/drone-detection -p data --unzip
```

3. After unzipping, you should have a **YOLO-style** layout: a root folder containing **`train`**, **`valid`** (or `val`), and **`test`**, each with **`images/`** and **`labels/`** subfolders. The Kaggle API may extract the dataset into a subfolder like `data/drone-detection-new.v5-new-train.yolov8/`. The example config expects the splits directly under `data/`. Move the `train`, `valid`, and `test` folders directly to `data/`, so your structure looks like this:

```text
data/
├── train/
├── valid/
└── test/
```

For example (adjust the folder name if Kaggle changes it):
```bash
mv data/drone-detection-new.v5-new-train.yolov8/* data/
rmdir data/drone-detection-new.v5-new-train.yolov8
```

**YOLO-ready batches:** `DroneDetectionYoloDataModule` resizes images to a square (default **640×640**), scales pixel values to **\[0, 1\]**, and parses each label file into YOLO boxes **in the same normalized format, relative to that square**. Each batch is a dict:

- `images`: tensor `(B, 3, H, W)`
- `targets`: length-`B` list of `(n_i, 5)` tensors with columns `cls, x_center, y_center, width, height`

Use `DroneDetectionYoloDataModule` from your own `LightningModule` or training script, for example:

```python
from pathlib import Path
from src.datasets.drone_detection_yolo import DroneDetectionYoloDataModule

dm = DroneDetectionYoloDataModule(Path("data/drone-detection"), batch_size=8)
# batch = next(iter(dm.train_dataloader()))
```

If your export uses **`val`** instead of **`valid`**, pass `val_dir="val"` into `DroneDetectionYoloDataModule` (or rename the folder on disk).

---

## Key abstractions

### `LightningModule` (`src/models/`)
Encapsulates the model architecture, loss function, optimizer, and train/val/test step logic. Adding a new model means writing a new `LightningModule` — `train_model.py` does not need to change.

### `LightningDataModule` (`src/datasets/`)
Encapsulates the dataset, splits, transforms, and dataloaders. Adding a new dataset means writing a new `LightningDataModule` — again, `train_model.py` does not need to change.

### Fiddle configs (`src/config/`)
Each config file is a Python module that returns a `fdl.Config[ExperimentConfig]`. Configs can inherit from a base config and override individual values — see `src/config/lr_comparison/` for an example. At training time, Fiddle resolves the full config and serialises it so the exact setup used in each run is reproducible.

**Important:** every Python object in a config must be wrapped in `fdl.Config` (or `fdl.Partial`), not instantiated directly. Fiddle's codegen only generates imports for objects it tracks through these wrappers — a plain instantiation will be reproduced in the generated file without its import, causing a `NameError` when the config is loaded.

---

## Running an experiment

```bash
python scripts/train_model.py src/config/convnet_utkface_classifier.py
```

To disable W&B logging:

```bash
python scripts/train_model.py src/config/convnet_utkface_classifier.py --no_wandb
```

To resume a previous run (e.g. after an early cancellation, power outage, or an HPC job timeout):

```bash
python scripts/train_model.py --resume_run_name <run_name>
```
