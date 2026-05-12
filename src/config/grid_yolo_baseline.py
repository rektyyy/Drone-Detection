import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.drone_detection_yolo import DroneDetectionYoloDataModule
from src.models.architectures.grid_convolutional import GridConvNetBackbone
from src.models.grid_yolo_baseline import GridYoloBaseline

def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 15
    image_size = 128
    num_classes = 3
    # 128 / (2^4 downsampling steps) = 8
    grid_size = 8
    
    channel_dims = [16, 32, 64, 128]
    backbone_out_channels = channel_dims[-1]

    architecture = fdl.Config(
        GridConvNetBackbone,
        input_shape=(3, image_size, image_size),
        channel_dims=channel_dims,
    )

    data_module = fdl.Config(
        DroneDetectionYoloDataModule,
        root="data",
        batch_size=16,
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
        mode="max",
        every_n_epochs=1,
        save_top_k=1,
    )

    model = fdl.Config(
        GridYoloBaseline,
        backbone=architecture,
        backbone_out_channels=backbone_out_channels,
        grid_size=grid_size,
        num_classes=num_classes,
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        name="grid_yolo_baseline",
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
