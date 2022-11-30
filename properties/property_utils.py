from models.base_model import BaseModel
from omegaconf import DictConfig
import pytorch_lightning as pl


class Property:
    """Property base class defining the structure of a property object.

    Args:
        logging_prefix (str): string to add to logging metric
    """

    def __init__(self, logging_prefix: str):
        self.logging_prefix = logging_prefix
        return

    def measure(
        self,
        config: DictConfig,
        model: BaseModel,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
    ):
        return
