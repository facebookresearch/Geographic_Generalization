from models.classifier_model import ClassifierModule
from omegaconf import DictConfig
import pytorch_lightning as pl
from abc import ABC, abstractmethod
from hydra.utils import instantiate


class Property(ABC):
    """Property base class defining the structure of a property object.

    Args:
        logging_name (str): name used in wandb logging (without prefix or suffix)
    Return:
        dict in the form {str: float}, where each key represents the name of the property, and each float is the corresponding value.
    """

    def __init__(self, logging_name: str, dataset_names: list[str]):
        self.logging_name = logging_name
        self.dataset_names = dataset_names
        return

    @abstractmethod
    def measure(
        self,
        config: DictConfig,
        model: ClassifierModule,
        trainer: pl.Trainer,
    ):
        return {self.logging_name: 0}


class DummyExample(Property):
    def __init__(self, logging_name: str, dataset_names: list[str]):
        super().__init__(logging_name=logging_name, dataset_names=dataset_names)

    def measure(
        self,
        config: DictConfig,
        model: ClassifierModule,
        trainer: pl.Trainer,
    ):
        # Set up a dict to store measurements
        results = {}

        # Make a datamodule
        datamodule_config = getattr(config, self.dataset_names[0])
        datamodule = instantiate(datamodule_config)

        # Perform the Measurement. You can do this a few main ways:
        #      1) manually iterate over datamodule
        #      2) call trainer.validate for basic accuracy measure
        res = trainer.validate(model=model, datamodule=datamodule)[0]
        # first result selected for simplicity

        # Log the results in wandb logger (in trainer) and in your results dict, which will be added to
        # the main CSV
        for metric in res.keys():
            trainer.logger.experiment.log(
                {self.logging_name + "_" + metric: res[metric]}
            )
            results.update({self.logging_name + "_" + metric: res[metric]})

        return results  # to be added to CSV
