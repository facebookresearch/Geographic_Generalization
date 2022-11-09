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

log = logging.getLogger(__name__)
git_hash = get_git_hash()


@hydra.main(
    version_base="1.2", config_path="config", config_name="experiment/example_experiment.yaml"
)
def main(config: DictConfig) -> None:
    # Set up 
    print_config(config)
    pl.seed_everything(config.seed)
    wandb_logger = setup_wandb(config, log, git_hash)
    job_logs_dir = os.getcwd()

    trainer = pl.Trainer(
        **config.trainer,
        plugins=SLURMEnvironment(auto_requeue=False),
        logger=wandb_logger,
    )

    for model in config.models: 
        # add loading a checkpoint 
        model = instantiate(model)

        for epoch in config.max_epochs: 
            measure_properties(property_configs = config.properties, model = model, wandb_logger = wandb_logger)
            evaluate_tasks(task_configs = config.tasks, model = model, wandb_logger = wandb_logger)

    # allows for logging separate experiments with multi-run (-m) flag
    wandb_logger.experiment.finish()

# Creates property objects defined in configs and measures them for the given model / logger
def measure_properties(property_configs: list, model: BaseModel, wandb_logger: WandbLogger):
    for property_config in property_configs: 
        prop = instantiate(property_config)
        measure = prop.measure(model, wandb_logger)

# Creates task objects defined in configs and measures them for the given model / logger
def evaluate_tasks(task_configs: list, model: BaseModel, wandb_logger: WandbLogger):
    for task_config in task_configs: 
        task = instantiate(task_config)
        results = task.evaluate(model, wandb_logger)

if __name__ == "__main__":
    user = os.getlogin()
    print("git hash: ", git_hash)
    snapshot_dir = tempfile.mkdtemp(prefix=f"/checkpoint/{user}/tmp/")
    print("Snapshot dir is: ", snapshot_dir)
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()
