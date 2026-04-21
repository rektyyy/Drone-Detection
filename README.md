# Neural Networks. Theory and Practice. 2026 Project Template

A template for training neural networks with [PyTorch Lightning](https://lightning.ai/), [Fiddle](https://github.com/google/fiddle), and [Weights & Biases](https://wandb.ai/).

The purpose of this template is to introduce PyTorch Lightning, Fiddle, and Weights & Biases together and showcase how they interact — not to provide a comprehensive guide to any of them individually. Each tool has far more functionality than is used here; the goal is to give you a working starting point that demonstrates a clean way to combine them.

The models included are not intended to be state-of-the-art. They exist solely to illustrate the abstractions that Lightning provides — specifically how a `LightningModule` encapsulates training logic and a `LightningDataModule` encapsulates data preparation, so that `train_model.py` stays model-agnostic and can train any experiment without modification.

---

## Installation

Create and activate a virtual environment using your preferred tool (e.g. `venv`, `conda`, or `micromamba`), then install the project. Example using `micromamba`:

```bash
micromamba create -n nn_2026_template python=3.12
micromamba activate nn_2026_template
pip install -e .
```

The `-e` flag installs the project in **editable mode** — instead of copying the source into `site-packages`, Python points directly at the `src/` directory. This means any changes you make to the source code take effect immediately without reinstalling.

All dependencies are declared in `pyproject.toml` and will be installed automatically.

### PyTorch and CUDA

`pyproject.toml` has no way to detect whether a GPU is present, so it cannot choose the right PyTorch build automatically. If you need a specific build, install PyTorch manually **before** running `pip install -e .`, and the subsequent install will skip it since it is already satisfied:

```bash
# CPU-only (Linux/Windows — macOS installs CPU by default)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

The available index URLs for other CUDA versions are listed at [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/).

```bash
micromamba install -c conda-forge libcst
```

---

## Project structure

```
.
├── data/                        # Datasets (see note below)
│   ├── mnist/
│   └── utkface/
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
    │   ├── convnet_mnist_classifier.py
    │   ├── convnet_utkface_classifier.py
    │   ├── convnet_utkface_regressor.py
    │   ├── mlp_utkface_classifier.py
    │   └── lr_comparison/       # Example of Fiddle config inheritance (learning rate sweep)
    │       ├── 1e-2.py
    │       └── 1e-4.py
    │
    ├── datasets/
    │   ├── mnist.py             # LightningDataModule for MNIST
    │   └── utkface.py           # LightningDataModule for UTKFace
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
ln -s /path/to/your/data/mnist data/mnist

# Windows (run as Administrator) — entire directory
mklink /D data C:\path\to\your\data

# Windows (run as Administrator) — individual datasets
mklink /D data\utkface C:\path\to\your\data\utkface
mklink /D data\mnist C:\path\to\your\data\mnist
```

This keeps all paths in the codebase relative (e.g. `data/utkface`), so configs and scripts work the same way on every machine regardless of where the data is physically stored.

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
python scripts/train_model.py src/config/convnet_mnist_classifier.py
```

To disable W&B logging:

```bash
python scripts/train_model.py src/config/convnet_mnist_classifier.py --no_wandb
```

To resume a previous run (e.g. after an early cancellation, power outage, or an HPC job timeout):

```bash
python scripts/train_model.py --resume_run_name <run_name>
```
