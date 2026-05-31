import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.drone_detection_yolo import DroneDetectionYoloDataModule
from src.models.yolov8_model import UltralyticsDetectionModel


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 30
    image_size = 640
    num_classes = 3

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
        mode="max",
        every_n_epochs=1,
        save_top_k=1,
    )

    model = fdl.Config(
        UltralyticsDetectionModel,
        model_name="yolov8n",
        num_classes=num_classes,
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        name="yolov8_baseline",
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
