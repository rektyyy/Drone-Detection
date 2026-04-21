import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import RandomHorizontalFlip

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.utkface import UTKFaceDataModule
from src.models.architectures.mlp import MLPBackbone
from src.models.classification_model import ClassificationModel


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 5
    embed_dim = 128
    image_size = 32

    architecture = fdl.Config(
        MLPBackbone,
        input_shape=(3, image_size, image_size),
        hidden_dims=[512, 256],
        output_dim=embed_dim,
        dropout=0.2,
    )

    data_module = fdl.Config(
        UTKFaceDataModule,
        "data/utkface",
        batch_size=128,
        size=image_size,
        transform=fdl.Config(RandomHorizontalFlip),
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
        ClassificationModel,
        architecture,
        embed_dim=embed_dim,
        num_classes=5,
        attribute="race",
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        "mlp_utkface_race_classifier",
        model,
        data_module,
        training_cfg=fdl.Config(
            TrainingConfig,
            wandb_logger,
            checkpoints_callback,
            max_epochs,
            callbacks=[],
        )
    )
