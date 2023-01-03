from properties.property_utils import Property
from models.base_model import BaseModel
from omegaconf import DictConfig
import pytorch_lightning as pl


class DCI(Property):
    """Example measure of a disentanglement to be completed"""

    def __init__(self, logging_name: str, dataset_names: list[str]):
        super().__init__(logging_name, dataset_names)

    def measure(
        self,
        config: DictConfig,
        model: BaseModel,
        trainer: pl.Trainer,
    ):
        trainer.logger.experiment.log({self.logging_name: 5})
        return 5
