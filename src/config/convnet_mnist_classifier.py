import fiddle as fdl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.config.schemas import ExperimentConfig, TrainingConfig
from src.datasets.mnist import MNISTDataModule
from src.models.architectures.convolutional import ConvNetBackbone
from src.models.classification_model import ClassificationModel


def build_config() -> fdl.Config[ExperimentConfig]:
    max_epochs = 5
    embed_dim = 128

    architecture = fdl.Config(
        ConvNetBackbone,
        input_shape=(1, 28, 28),
        channel_dims=[32, 64],
        output_dim=embed_dim,
        dropout=0.,
    )

    data_module = fdl.Config(
        MNISTDataModule,
        "data/mnist",
        batch_size=128,
    )

    wandb_logger = fdl.Partial(
        WandbLogger,
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
    )

    checkpoints_callback = fdl.Partial(
        ModelCheckpoint,
        monitor="val/acc",
        every_n_epochs=1,
        save_top_k=1,
        mode="max",
    )

    model = fdl.Config(
        ClassificationModel,
        architecture,
        embed_dim=embed_dim,
        num_classes=10,
        attribute="label",
        lr=1e-3,
    )

    return fdl.Config(
        ExperimentConfig,
        "convnet_mnist_classifier",
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
