import hydra
from submitit.helpers import RsyncSnapshot
from hydra.utils import instantiate
from omegaconf import DictConfig
from models.loggers import (
    setup_wandb,
    print_config,
    get_git_hash,
    find_existing_checkpoint,
)
import os
import pytorch_lightning as pl
import tempfile
from pytorch_lightning.plugins.environments import SLURMEnvironment
import logging
from models.base_model import BaseModel
from models.loggers import WandbLogger

log = logging.getLogger(__name__)
git_hash = get_git_hash()


@hydra.main(
    version_base="1.2", config_path="config", config_name="evaluate_defaults.yaml"
)
def main(config: DictConfig) -> None:
    ## Set up
    print_config(config)
    pl.seed_everything(config.seed)
    wandb_logger = setup_wandb(config, log, git_hash)
    model = instantiate(config.module)

    # Run experiment functions
    measure_properties(config=config, model=model, wandb_logger=wandb_logger)
    evaluate_tasks(config=config, model=model, wandb_logger=wandb_logger)

    # allows for logging separate experiments with multi-run (-m) flag
    wandb_logger.experiment.finish()


# Creates property objects defined in configs and measures them for the given model / logger
def measure_properties(config: DictConfig, model: BaseModel, wandb_logger: WandbLogger):
    properties = config.properties
    datamodule = instantiate(config.datamodule)

    for property_name in properties:
        property_config = getattr(config, property_name)
        print(f"Builiding property config: {property_name}")
        property = instantiate(property_config)

        property.measure(model, datamodule, wandb_logger)


# Creates task objects defined in configs and measures them for the given model / logger
def evaluate_tasks(config: DictConfig, model: BaseModel, wandb_logger: WandbLogger):
    tasks = config.tasks
    for task_name in tasks:
        task_config = getattr(config, task_name)
        print(f"Builiding task config: {task_name}")
        task = instantiate(task_config)
        task.evaluate(model, wandb_logger)


if __name__ == "__main__":
    user = os.getlogin()
    print("git hash: ", git_hash)
    snapshot_dir = tempfile.mkdtemp(prefix=f"/checkpoint/{user}/tmp/")
    print("Snapshot dir is: ", snapshot_dir)
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()
