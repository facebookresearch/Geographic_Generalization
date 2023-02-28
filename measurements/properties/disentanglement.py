from measurement_utils import Measurement
from models.classifier_model import ClassifierModule
from omegaconf import DictConfig


class DCI(Measurement):
    """Example measure of a disentanglement measure to be completed"""

    def __init__(
        self,
        datamodule_names: list[str],
        model: ClassifierModule,
        config: DictConfig,
    ):
        super().__init__(datamodule_names, model, config)

    def measure(
        self,
    ) -> dict[str:float]:
        datamodule_name, datamodule = next(iter(self.datamodules.items()))
        property_name = "dci"
        return {f"{datamodule_name}_test_{property_name}": 0.1}