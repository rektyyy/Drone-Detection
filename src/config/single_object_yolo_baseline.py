import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.drone_detection_yolo import DroneDetectionYoloDataModule
from src.models.architectures.convolutional import ConvNetBackbone
from src.models.single_object_yolo_baseline import SingleObjectYoloBaseline


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 10
    embed_dim = 128
    image_size = 256
    num_classes = 3

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
        monitor="val/mAP50",
        mode="max", # We want to maximize mAP50
        every_n_epochs=1,
        save_top_k=1,
    )

    model = fdl.Config(
        SingleObjectYoloBaseline,
        backbone=architecture,
        embed_dim=embed_dim,
        num_classes=num_classes,
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        name="single_object_yolo_baseline",
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
