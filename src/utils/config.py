import runpy
from pathlib import Path

import fiddle as fdl
import wandb


def parse_fiddle_config(config_path: str) -> fdl.Config:
    config_vars = runpy.run_path(config_path)
    if "build_config" not in config_vars:
        raise ValueError(
            "The provided .py file should define a function named `build_config`."
        )
    config = config_vars["build_config"]

    return config()

def get_wandb_config(run_path: str, artifact_type: str = "config") -> fdl.Config:
    api = wandb.Api()
    run = api.run(run_path)

    if len(run.logged_artifacts()) == 0:
        raise ValueError(f"No artifacts found for run {run_path}")

    cfg_artifact = None
    for artifact in run.logged_artifacts():
        if artifact.type == artifact_type:
            cfg_artifact = artifact
            break

    if cfg_artifact is None:
        raise ValueError(f"No config artifact found for run {run_path}")

    cfg_dir = cfg_artifact.download()
    cfg_dir = Path(cfg_dir)
    for cfg_file in cfg_dir.glob("*.py"):
        cfg = parse_fiddle_config(str(cfg_file))
        return cfg

    raise ValueError(f"No config file found for run {run_path}")
