from measurement_utils import Measurement
from models.classifier_model import ClassifierModule
from omegaconf import DictConfig
import pytorch_lightning as pl


class Equivariance(Measurement):
    """Measures equivariance of model embeddings with respect to augmentations"""

    def __init__(self, logging_name: str, dataset_names: list[str]):
        super().__init__(logging_name, dataset_names)

    def test_step(self, batch, batch_idx):
        x, labels = batch
        z = self(x)

        for magnitude_idx in range(10):
            transform = transformations.Transformation(
                self.transform_name, magnitude_idx
            )
            x_t = transform(x)
            z_t = self(x_t)
            d = z - z_t
            self.magnitude_to_diff[magnitude_idx].append(d.cpu())

    def measure(
        self,
        config: DictConfig,
        model_config: dict,
    ) -> dict[str:float]:
        return {self.logging_name: 0}
