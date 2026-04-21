import warnings
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import click
import fiddle as fdl
import lightning as L
from fiddle.codegen import codegen
from fiddle.printing import as_dict_flattened

import wandb
from src.config.constants import WANDB_ENTITY, WANDB_PROJECT
from src.utils.config import get_wandb_config, parse_fiddle_config

if TYPE_CHECKING:
    from lightning.pytorch.loggers import Logger

    from src.config.schemas import ExperimentConfig

@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, file_okay=True), required=False)
@click.option("--resume_run_name", type=str, default=None)
@click.option("--no_wandb", is_flag=True, default=False)
@click.option("--seed", type=int, default=42)
def main(config_path, resume_run_name, no_wandb, seed):
    L.seed_everything(seed)

    if resume_run_name is not None:
        # Fetch the config from the original run stored in W&B — ensures the resumed run
        # uses the exact same config. config_path is ignored.
        cfg: fdl.Config[ExperimentConfig] = get_wandb_config(f"{WANDB_ENTITY}/{WANDB_PROJECT}/{resume_run_name}")
        run_name = resume_run_name
    else:
        # Config files are Python modules that must be imported dynamically —
        # parse_fiddle_config handles this.
        cfg: fdl.Config[ExperimentConfig] = parse_fiddle_config(config_path)
        # Suffix with timestamp to ensure the run name is unique across runs of the same config.
        run_name = cfg.name + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    log_dir = f"logs/{run_name}"

    ckpt_path = None
    # Find the latest checkpoint to resume training from, preferring last.ckpt —
    # Lightning's dedicated file that always points to the most recent epoch.
    if resume_run_name is not None:
        last_ckpt = Path(log_dir) / "last.ckpt"
        if last_ckpt.exists():
            ckpt_path = str(last_ckpt)
        else:
            # last.ckpt not found — fall back to the most recently modified checkpoint.
            ckpts = sorted(Path(log_dir).glob("*.ckpt"), key=lambda p: p.stat().st_mtime)
            if ckpts:
                ckpt_path = str(ckpts[-1])
        if ckpt_path is None:
            warnings.warn(f"No checkpoint found in {log_dir}")

    built_cfg: ExperimentConfig = fdl.build(cfg)
    # model is a LightningModule — it encapsulates the architecture, loss function,
    # optimizer, and train/val/test step logic, keeping this script model-agnostic.
    model: L.LightningModule = built_cfg.model

    # wandb_logger and checkpoint_callback are stored as partials in the config because
    # they require run_name and log_dir, which are only known at runtime
    partial_checkpoint_callback = built_cfg.training_cfg.checkpoint_callback
    checkpoint_callback = partial_checkpoint_callback(dirpath=log_dir) if partial_checkpoint_callback is not None else None
    callbacks = built_cfg.training_cfg.callbacks + ([checkpoint_callback] if checkpoint_callback is not None else [])

    logger: list[Logger] = []
    partial_wandb_logger = built_cfg.training_cfg.wandb_logger
    if partial_wandb_logger is not None and not no_wandb:
        # id ties this run to its W&B entry — required for resumption.
        # resume="allow" means W&B will start a fresh run if no run with this id exists yet.
        wandb_logger = partial_wandb_logger(
            id=run_name,
            name=run_name,
            resume="allow",
            tags=[f"seed: {seed}"],
        )
        logger.append(wandb_logger)

        # Config is only uploaded on new runs — it was already uploaded during the original run.
        if config_path is not None and resume_run_name is None:
            # Upload config values as W&B run config — populates the hyperparameter columns
            # in the runs table, enabling filtering and side-by-side comparison.
            wandb_logger.experiment.config.update(
                as_dict_flattened(cfg)
            )
            # fiddle.codegen serializes the fully resolved config — all values are inlined,
            # including those inherited from base configs. This ensures the uploaded file is
            # self-contained and accurately reflects what was actually used in this run.
            # Uploading the file at config_path directly would only capture the overrides,
            # losing any values inherited from a base config (e.g. if config_path points to
            # src/config/lr_comparison/1e-2.py, all values from convnet_utkface_classifier.py
            # would be lost).
            generated = codegen.codegen_dot_syntax(cfg)
            code_str = "\n".join(generated.lines())
            artifact = wandb.Artifact(name=f"{run_name}_config", type="config")
            with artifact.new_file("config.py", mode="w") as file:
                file.write(code_str)

            wandb_logger.experiment.log_artifact(artifact)

    # data_module is a LightningDataModule — it encapsulates the dataset, train/val/test
    # splits, transforms, and dataloaders, keeping data preparation self-contained.
    data_module: L.LightningDataModule = built_cfg.data_module

    trainer = L.Trainer(
        log_every_n_steps=50,
        logger=logger,
        max_epochs=built_cfg.training_cfg.max_epochs,
        default_root_dir=log_dir,
        callbacks=callbacks,
    )
    trainer.fit(
        model=model,
        datamodule=data_module,
        ckpt_path=ckpt_path,
    )


if __name__ == "__main__":
    main()
