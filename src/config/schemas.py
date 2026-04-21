from dataclasses import dataclass
from functools import partial

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


@dataclass
class TrainingConfig:
    wandb_logger: partial[WandbLogger] | None
    checkpoint_callback: partial[ModelCheckpoint] | None
    max_epochs: int
    callbacks: list[L.Callback]


@dataclass
class ExperimentConfig:
    name: str
    model: L.LightningModule
    data_module: L.LightningDataModule
    training_cfg: TrainingConfig
