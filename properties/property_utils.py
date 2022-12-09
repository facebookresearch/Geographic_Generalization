from models.base_model import BaseModel
from omegaconf import DictConfig
import pytorch_lightning as pl
from abc import ABC, abstractmethod

class Property(ABC):
    """Property base class defining the structure of a property object.

    Args:
        logging_name (str): name used in wandb logging 
    """

    def __init__(self, logging_name: str):
        self.logging_name = logging_name
        return

    @abstractmethod
    def measure(
        self,
        config: DictConfig,
        model: BaseModel,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
    ):
        return
