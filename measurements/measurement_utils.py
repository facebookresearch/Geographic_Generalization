"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from omegaconf import DictConfig
from abc import ABC, abstractmethod
from models.classifier_model import ClassifierModule
from hydra.utils import instantiate
from datasets.image_datamodule import ImageDataModule
import pytorch_lightning as pl
from pytorch_lightning.plugins.environments import SLURMEnvironment
import os
import pandas as pd


class Measurement(ABC):
    """Measure base class defining the structure of a measure object.

    This base class constructor creates the following objects for use:
    1) self.datamodules: a dictionary of datamodules, indexed by the datamodule's name (e.g. {datamodule_name : datamodule}). This dictionary contains a datamodule for each value passed in the parameter 'datamodule_names', so will *only* include datamodules for
        those datamodules. To access a datamodule, simply index the self.datamodules dictionary with datamodule name you want (again, this key must be a value in datamodule_names parameter). Example: self.datamodules['imagenet'].

    2) self.model: the instantiated model object to use in the measurement.

    3) self.trainer: the trainer you can use to evaluate the model.

    The base class also creates the following functions:
    1) self.save_extra_results_to_csv: this function takes in any dictionary, makes a folder with the measurement class name, and saves the dictionary as a CSV. It can be used to store model predictions or other measurement details.

    Args:
        datamodule_names (list[str]): list of datamodule names required for this measurement. E.g. ['imagenet', 'dollarstreet']
        model (ClassifierModule): pytorch model to perform the measurement with
        experiment_config (DictConfig): Hydra config used primarily to instantiate a trainer. Must have key: 'trainer' to be compatible with pytorch lightning.
    Return:
        dict in the form {str: float}, where each key represents the name of the measurement, and each float is the corresponding value. Keys should be in the form: <datamodule_name>_<split>_<property_name>, all lowercase (e.g. imagenet_test_accuracy).
    """

    def __init__(
        self,
        datamodule_names: list[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
    ):
        self.datamodules = self.make_datamodules(experiment_config, datamodule_names)
        self.trainer = self.make_trainer(experiment_config=experiment_config)
        self.experiment_config = experiment_config
        self.save_detailed_results = experiment_config.save_detailed_results
        self.model = model
        self.datamodule_names = datamodule_names
        return

    def make_trainer(self, experiment_config: DictConfig):
        trainer = pl.Trainer(
            **experiment_config.trainer,
            plugins=SLURMEnvironment(auto_requeue=False),
        )
        return trainer

    def make_datamodules(
        self, experiment_config: DictConfig, datamodule_names: list[str]
    ) -> dict[str, ImageDataModule]:
        datamodules = {}
        for datamodule_name in datamodule_names:
            print(experiment_config.keys())
            datamodule_config = getattr(experiment_config, datamodule_name)
            datamodule = instantiate(datamodule_config)
            datamodules[datamodule_name] = datamodule

        return datamodules

    def save_extra_results_to_csv(self, extra_results: dict[str, list], name: str):
        print(self.save_detailed_results)
        if self.save_detailed_results:
            measurement_folder = self.__class__.__name__
            os.makedirs(measurement_folder, exist_ok=True)
            save_path = f"{measurement_folder}/{name}.csv"
            pd.DataFrame(extra_results).to_csv(save_path)
        else:
            print(
                "\n\n save_extra_results was called, but did not run because the configuration parameter 'save_detailed_results' is set to False. If you'd like to save detailed results, change the 'save_detailed_results' parameter in config/mode to 'True'."
            )
        return

    @abstractmethod
    def measure(
        self,
    ) -> dict[str, float]:
        datamodule_name, datamodule = next(iter(self.datamodules.items()))
        property_name = "property"

        return {f"{datamodule_name}_{property_name}": 0.1}
