import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.drone_detection_yolo import DroneDetectionYoloDataModule
from src.models.architectures.grid_convolutional import GridConvNetBackbone
from src.models.yolov1_baseline import YoloV1Baseline


def build_config() -> fdl.Config[ExperimentConfig]:
    """Minimal smoke-test config: 1 epoch, tiny backbone, small batches.

    Runs the full pipeline (train -> val -> test + W&B logging) in a few minutes
    on CPU. Not meant to produce useful weights — use to verify code changes before
    launching a real training run.
    """
    image_size = 128
    num_classes = 3
    grid_size = 8  # 128 / 2^4

    channel_dims = [8, 16, 32, 64]  # half the baseline widths
    backbone_out_channels = channel_dims[-1]

    architecture = fdl.Config(
        GridConvNetBackbone,
        input_shape=(3, image_size, image_size),
        channel_dims=channel_dims,
    )

    data_module = fdl.Config(
        DroneDetectionYoloDataModule,
        root="data",
        batch_size=4,
        img_size=image_size,
        num_workers=0,  # avoid spawning workers for a quick test
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
        YoloV1Baseline,
        backbone=architecture,
        backbone_out_channels=backbone_out_channels,
        grid_size=grid_size,
        num_classes=num_classes,
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        name="yolov1_smoke",
        model=model,
        data_module=data_module,
        training_cfg=fdl.Config(
            TrainingConfig,
            wandb_logger=wandb_logger,
            checkpoint_callback=checkpoints_callback,
            max_epochs=1,
            callbacks=[],
        )
    )
