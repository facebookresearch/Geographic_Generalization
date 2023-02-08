from __future__ import annotations  # this is to call 'list
from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
import pytorch_lightning as pl
import torch
from models.classifier_model import ClassifierModule
from torch import Tensor


class Sparsity(Measurement):
    """Measures sparsity of model embeddings as the number of fired neurons per sample.

    Assumes model has forward_features() function
    """

    def __init__(
        self,
        dataset_names: list[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
        threshold: float = None,
    ):  # what if I want to put threshold in experiment_config, it will raise an error when initialize the super()
        super().__init__(dataset_names, model, experiment_config)
        self.model = model
        self.model.test_step = self.test_step
        self.z = torch.empty(0)
        self.threshold = threshold

    def reset_stored_z(self):
        self.z = torch.empty(0)

    def test_step(self, batch, batch_idx):
        x, _ = batch
        # if we make forward features all layers, we could play with the layer we compute metrics on
        z = self.model.forward_features(x)
        self.z = torch.cat([self.z, z])
        return None

    @staticmethod
    def measure_sparsity(z: Tensor, threshold: float = 0.1) -> float:
        """
        Args:
            z: (num samples, embedding dim)
        """
        theta = threshold * torch.ones_like(z)

        active_neurons = (z.abs() > theta).float()
        sparsity = active_neurons.mean()

        return sparsity

    def measure(self):

        # Get datamodule of interest
        datamodule_name, datamodule = next(iter(self.datamodules.items()))
        self.reset_stored_z()

        gpus = 1 if torch.cuda.is_available() else 0
        self.trainer.test(
            self.model,
            datamodule=datamodule,
        )

        sparsity = self.measure_sparsity(
            self.z, self.threshold
        )
        results = {
            f"{datamodule_name}_sparsity": sparsity,
        }
        return results
