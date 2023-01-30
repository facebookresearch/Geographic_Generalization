from measurement_utils import Measurement
from models.classifier_model import ClassifierModule
from omegaconf import DictConfig
import pytorch_lightning as pl


class DCI(Measurement):
    """Example measure of a disentanglement measure to be completed"""

    def __init__(
        self,
        logging_name: str,
        dataset_names: list[str],
        model: ClassifierModule,
        config: DictConfig,
    ):
        super().__init__(logging_name, dataset_names, model, config)

    def measure(
        self,
    ) -> dict[str:float]:
        return {self.logging_name: 0}
