import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torchvision.transforms import RandomHorizontalFlip

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.utkface import UTKFaceDataModule
from src.models.architectures.convolutional import ConvNetBackbone
from src.models.regression_model import RegressionModel


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 5
    embed_dim = 128
    image_size = 32

    architecture = fdl.Config(
        ConvNetBackbone,
        input_shape=(3, image_size, image_size),
        channel_dims=[32, 64, 128],
        output_dim=embed_dim,
        dropout=0.,
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
        monitor="val/mae",
        every_n_epochs=1,
        save_top_k=1,
        mode="min",
    )

    model = fdl.Config(
        RegressionModel,
        architecture,
        embed_dim=embed_dim,
        attribute="age",
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        "convnet_utkface_age_regressor",
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
