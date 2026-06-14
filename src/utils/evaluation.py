"""Helpers for loading trained checkpoints and running test-set evaluation.

Used by notebooks/model_comparison.ipynb to evaluate several models on the same
test split without the W&B logging machinery of scripts/train_model.py.
"""

from pathlib import Path

import fiddle as fdl
import lightning as L
import torch

from src.utils.config import parse_fiddle_config


def find_best_checkpoint(run_glob: str, logs_dir: str = "logs") -> Path:
    """Return the most recent checkpoint matching ``run_glob`` under ``logs_dir``.

    Run directories are named ``<config_name>_<timestamp>``, so a glob like
    ``"yolov8_baseline_*/*.ckpt"`` selects every checkpoint of that config; the most
    recently modified one (the latest run's best checkpoint) is returned.
    """
    matches = sorted(Path(logs_dir).glob(run_glob), key=lambda p: p.stat().st_mtime)
    if not matches:
        raise FileNotFoundError(f"No checkpoint matching {run_glob!r} under {logs_dir}/")
    return matches[-1]


def load_model_from_config(config_path: str, checkpoint_path) -> L.LightningModule:
    """Rebuild a model from its Fiddle config and load trained weights from a checkpoint.

    The models don't call ``save_hyperparameters()``, so Lightning's
    ``load_from_checkpoint`` cannot reconstruct them. Instead we build a fresh model
    with the exact architecture from its config, then load the checkpoint's state dict.
    Returns the model in eval mode.
    """
    built: object = fdl.build(parse_fiddle_config(config_path))
    model: L.LightningModule = built.model
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def build_datamodule_from_config(config_path: str) -> L.LightningDataModule:
    """Build just the data module from a config (for evaluation at matching img_size)."""
    built: object = fdl.build(parse_fiddle_config(config_path))
    return built.data_module


def run_test_pass(model: L.LightningModule, datamodule: L.LightningDataModule) -> dict:
    """Run ``trainer.test`` on the test split and return the logged metrics dict.

    Uses a logger-free Trainer so it stays silent and side-effect free — no W&B run,
    no checkpoints written.
    """
    trainer = L.Trainer(logger=False, enable_checkpointing=False, accelerator="auto", devices=1)
    results = trainer.test(model=model, datamodule=datamodule, verbose=False)
    return results[0] if results else {}
