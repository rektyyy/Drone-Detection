import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.drone_detection_yolo import DroneDetectionYoloDataModule
from src.models.architectures.convolutional import ConvNetBackbone
from src.models.class_agnostic_dummy_model import ClassAgnosticDummyModel


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 5
    embed_dim = 32
    image_size = 128

    architecture = fdl.Config(
        ConvNetBackbone,
        input_shape=(3, image_size, image_size),
        channel_dims=[8, 16, 32, 64],
        output_dim=embed_dim,
        dropout=0.0,
    )

    data_module = fdl.Config(
        DroneDetectionYoloDataModule,
        root="data",
        batch_size=8,
        img_size=image_size,
    )

    wandb_logger = fdl.Partial(
        WandbLogger,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    checkpoints_callback = fdl.Partial(
        ModelCheckpoint,
        monitor="val/loss",
        every_n_epochs=1,
        save_top_k=1,
    )

    model = fdl.Config(
        ClassAgnosticDummyModel,
        backbone=architecture,
        embed_dim=embed_dim,
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        name="class_agnostic_dummy_baseline",
        model=model,
        data_module=data_module,
        training_cfg=fdl.Config(
            TrainingConfig,
            wandb_logger=wandb_logger,
            checkpoint_callback=checkpoints_callback,
            max_epochs=max_epochs,
            callbacks=[],
        )
    )
