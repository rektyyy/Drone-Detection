import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.drone_detection_yolo import DroneDetectionYoloDataModule
from src.models.trivial_baseline import TrivialBaseline


def build_config() -> fdl.Config[ExperimentConfig]:
    """Always-center-box trivial baseline.

    No learnable parameters are trained. Run for 1 epoch to get val/test mAP
    metrics that establish the lower bound for the comparison table.

    Prior statistics from training labels:
        most frequent class = Drone (1), mean box w=0.089, h=0.088 (relative).
    """
    data_module = fdl.Config(
        DroneDetectionYoloDataModule,
        root="data",
        batch_size=16,
        img_size=128,
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
        TrivialBaseline,
        prior_class=1,
        prior_w=0.089,
        prior_h=0.088,
        num_classes=3,
    )

    return fdl.Config(
        ExperimentConfig,
        name="trivial_baseline",
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
