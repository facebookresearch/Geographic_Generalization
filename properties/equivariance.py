from properties.property_utils import Property
from models.base_model import BaseModel
from omegaconf import DictConfig
import pytorch_lightning as pl


class Aug_Approximation(Property):
    """Example measure of an equivariance property to be completed"""

    def __init__(self, logging_prefix: str):
        super().__init__(logging_prefix)

    def measure(
        self,
        config: DictConfig,
        model: BaseModel,
        datamodule: pl.LightningDataModule,
        trainer: pl.Trainer,
    ):
        trainer.logger.experiment.log({self.logging_prefix: 13})
        return 1
