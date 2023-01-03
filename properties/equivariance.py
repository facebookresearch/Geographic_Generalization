from properties.property_utils import Property
from models.classifier_model import ClassifierModule
from omegaconf import DictConfig
import pytorch_lightning as pl


class AugmentationEquivariance(Property):
    """Example measure of an equivariance property to be completed"""

    def __init__(self, logging_name: str, dataset_names: list[str]):
        super().__init__(logging_name=logging_name, dataset_names=dataset_names)

    def measure(
        self,
        config: DictConfig,
        model: ClassifierModule,
        trainer: pl.Trainer,
    ):
        trainer.logger.experiment.log({self.logging_name: 13})
        return {self.logging_name: 13}
