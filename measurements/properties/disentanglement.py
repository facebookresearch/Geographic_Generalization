from measurement_utils import Measurement
from models.classifier_model import ClassifierModule
from omegaconf import DictConfig
import pytorch_lightning as pl


class DCI(Measurement):
    """Example measure of a disentanglement measure to be completed"""

    def __init__(
        self,
        dataset_names: list[str],
        model: ClassifierModule,
        config: DictConfig,
    ):
        super().__init__(dataset_names, model, config)

    def measure(
        self,
    ) -> dict[str:float]:
        dataset_name, datamodule = next(iter(self.datamodules.items()))
        property_name = "DCI"
        return {f"{dataset_name}_{property_name}": 0.1}
