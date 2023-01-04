from measurement_utils import Measurement
from models.classifier_model import ClassifierModule
from omegaconf import DictConfig
import pytorch_lightning as pl


class DCI(Measurement):
    """Example measure of a disentanglement measure to be completed"""

    def __init__(self, logging_name: str, dataset_names: list[str]):
        super().__init__(logging_name, dataset_names)

    def measure(
        self,
        config: DictConfig,
        model_config: dict,
    ):
        return {self.logging_name: 0}
