from models.base_model import BaseModel
from omegaconf import DictConfig
import pytorch_lightning as pl
from abc import ABC, abstractmethod

class Property(ABC):
    """Property base class defining the structure of a property object.

    Args:
        logging_name (str): name used in wandb logging (without prefix or suffix) 
    """

    def __init__(self, logging_name: str, dataset_names: list[str]):
        self.logging_name = logging_name
        self.dataset_names = dataset_names
        return

    @abstractmethod
    def measure(
        self,
        config: DictConfig,
        model: BaseModel,
        trainer: pl.Trainer,
    ):
        return
