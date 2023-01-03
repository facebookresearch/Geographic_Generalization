import hydra
from submitit.helpers import RsyncSnapshot
from hydra.utils import instantiate
from omegaconf import DictConfig
from models.loggers import (
    setup_wandb,
    print_config,
    get_git_hash,
)
import os
import pytorch_lightning as pl
import tempfile
from pytorch_lightning.plugins.environments import SLURMEnvironment
import logging
from models.classifier_model import ClassifierModule
import pandas as pd

log = logging.getLogger(__name__)
git_hash = get_git_hash()


@hydra.main(
    version_base="1.2", config_path="config", config_name="evaluate_defaults.yaml"
)
def main(config: DictConfig) -> None:
    print_config(config)
    pl.seed_everything(config.seed)
    wandb_logger = setup_wandb(config, log, git_hash)
    model = instantiate(config.module)
    trainer = pl.Trainer(
        **config.trainer,
        plugins=SLURMEnvironment(auto_requeue=False),
        logger=wandb_logger,
    )
    measurements = {}

    # Run experiment functions
    property_measurements = measure_properties(
        config=config, model=model, trainer=trainer
    )
    task_measurements = evaluate_tasks(config=config, model=model, trainer=trainer)

    # Combine measurement logs into larger dict
    measurements.update(property_measurements)
    measurements.update(task_measurements)

    # Make a dataframe, and save it
    measurements_dataframe = pd.DataFrame(measurements, index=[0])
    print(measurements_dataframe)
    print(f"{getattr(config, 'logs_dir')}/measurements.csv")
    measurements_dataframe.to_csv(f"{getattr(config, 'logs_dir')}/measurements.csv")

    wandb_logger.experiment.finish()


def measure_properties(
    config: DictConfig, model: ClassifierModule, trainer: pl.Trainer
):
    """Pulls property configs from config.properties list, builds property objects, and calls measure on each.

    Args:
        config (DictConfig): Hydra DictConfig with two main requirements:
                1) a 'properties' key which corresponds to a list of strings specifying the names of property objects to build
                2) each property name must be a key in config, which maps to the targets / hyperparameters for the property object (input for hydra's instantiate call)
                E.g. config = {
                    'properties': [property1],
                    'property1': {
                        __target__: example_path
                    }

                }

        model (ClassifierModule): Pytorch Lightning Module
        trainer (pl.Trainer): Pytorch Lightning Trainer
    """
    properties = config.properties
    property_measurements = {}

    for property_name in properties:
        print(f"\n\n *** Measuring Property : {property_name} *** \n\n")
        property_config = getattr(config, property_name)
        property = instantiate(property_config)
        measures = property.measure(config, model, trainer)
        property_measurements.update(measures)

    return property_measurements


# Creates task objects defined in configs and measures them for the given model / logger
def evaluate_tasks(config: DictConfig, model: ClassifierModule, trainer: pl.Trainer):
    """Pulls task configs from config.tasks list, builds task objects, and calls evaluate on each.

    Args:
        config (DictConfig): Hydra DictConfig with two main requirements:
                1) a 'tasks' key which corresponds to a list of strings specifying the names of task objects to build
                2) each task name must be a key in config, which maps to the targets / hyperparameters for the task object (input for hydra's instantiate call)
                E.g. config = {
                    'tasks': [task1],
                    'task1': {
                        __target__: example_path,
                        hyperparam: 5
                    }

                }
        model (ClassifierModule): Pytorch Lightning Module
        trainer (pl.Trainer): Pytorch Lightning Trainer
    """
    tasks = config.tasks
    task_measurements = {}
    for task_name in tasks:
        print(f"\n\n *** Starting Task : {task_name} *** \n\n")
        task_config = getattr(config, task_name)
        task = instantiate(DictConfig(task_config))
        eval_results = task.evaluate(config, model, trainer)
        task_measurements.update(eval_results)

    return task_measurements


if __name__ == "__main__":
    user = os.getlogin()
    print("git hash: ", git_hash)
    snapshot_dir = tempfile.mkdtemp(prefix=f"/checkpoint/{user}/tmp/")
    print("Snapshot dir is: ", snapshot_dir)
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()
