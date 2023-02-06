import hydra
from submitit.helpers import RsyncSnapshot
from hydra.utils import instantiate
from omegaconf import DictConfig
from models.loggers import (
    setup_wandb,
    print_config,
    get_git_hash,
)
import wandb
import os
import pytorch_lightning as pl
import tempfile
import logging
from models.classifier_model import ClassifierModule
import pandas as pd
from pytorch_lightning.loggers import WandbLogger
import copy

log = logging.getLogger(__name__)
git_hash = get_git_hash()


@hydra.main(
    version_base="1.2", config_path="config", config_name="evaluate_defaults.yaml"
)
def main(config: DictConfig) -> None:
    pl.seed_everything(config.seed)
    wandb_logger = setup_wandb(config, log, git_hash)

    # Build model
    model = instantiate(config.model)

    # Run experiment functions
    measurements = perform_measurements(
        model=model, experiment_config=config, wandb_logger=wandb_logger
    )

    # Make a dataframe, and save it
    measurements_dataframe = pd.DataFrame(measurements, index=[0])
    measurements_dataframe.to_csv(f"{getattr(config, 'logs_dir')}/measurements.csv")

    wandb_logger.experiment.finish()


def perform_measurements(
    model: ClassifierModule,
    experiment_config: DictConfig,
    wandb_logger: WandbLogger,
):
    """Pulls measurement configs from config.measurements list, builds measurement objects, and calls measure function on each.

    Args:
        model: instantiated model object following ClassifierModule class attributes
        config (DictConfig): Hydra DictConfig with two main requirements:
                1) a 'measurements' key which corresponds to a list of strings specifying the names of measurement objects to build
                2) each measurement name must be a key in config, which maps to the targets / hyperparameters for the measurement object (input for hydra's instantiate call)
                3) a 'model' key which maps to a hydra model config

                ** A correct config looks like this:
                        config = {
                            'model': resnet18
                            'measurement': [measure_name1, measure_name2],
                            'measure_name1': {
                                __target__: example_path1
                            }
                            'measure_name2': {
                                __target__: example_path2
                                hyperparam1: 5
                            }
                        }
                ** An incorrect config looks like this (missing an object for measure_name2):
                        config = {
                            'model': resnet18
                            'measurement': [measure_name1, measure_name2],
                            'measure_name1': {
                                __target__: example_path1
                            }
                        }

        wandb_logger (WandbLogger): wandb logger to keep track of the experiment results

    """
    measurement_names = experiment_config.measurements
    results = {}

    for measurement_name in measurement_names:
        print(f"\n\n *** Measuring : {measurement_name} *** \n\n")
        measurement_config = getattr(experiment_config, measurement_name)
        measurement = instantiate(
            measurement_config,
            model=copy.deepcopy(model),
            experiment_config=experiment_config,
            _recursive_=False,
        )
        result = measurement.measure()

        # log results in measurement subsection
        wandb.log({f"{measurement_name}/{k}": result[k] for k in list(result.keys())})
        results.update(result)

    return results


if __name__ == "__main__":
    user = os.getlogin()
    snapshot_dir = tempfile.mkdtemp(prefix=f"/checkpoint/{user}/tmp/")
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()
