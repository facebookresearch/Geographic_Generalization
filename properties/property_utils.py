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
