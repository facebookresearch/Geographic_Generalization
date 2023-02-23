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
        datamodule_names: list[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
        thresholds: float = [0.05, 0.1, 0.25, 0.5, 1, 1.5, 2, 2.5, 3],
    ):  # what if I want to put threshold in experiment_config, it will raise an error when initialize the super()
        super().__init__(datamodule_names, model, experiment_config)
        self.model.test_step = self.test_step
        self.z = torch.empty(0)
        self.thresholds = thresholds  # Right now, threshold is an attribute of Sparsity object, but we can also pass it as argument to measure_sparsity()

    def reset_stored_z(self):
        self.z = torch.empty(0)

    def test_step(self, batch, batch_idx):
        if len(batch) == 2:
            x, _ = batch
        elif len(batch) == 3:
            x, _, _ = batch
        # if we make forward features all layers, we could play with the layer we compute metrics on
        z = self.model.forward_features(x)
        self.z = torch.cat([self.z.to(self.model.device), z])

        return None

    @staticmethod
    def measure_sparsity(z: Tensor, threshold: float) -> float:
        """
        Args:
            z: (num samples, embedding dim)
        """
        theta = threshold * torch.ones_like(z)
        active_neurons = (z.abs() > theta).float()
        sparsity = active_neurons.mean()

        return sparsity.item()

    def measure(self):
        # Get datamodule of interest
        datamodule_name, datamodule = next(iter(self.datamodules.items()))

        results = {}

        for threshold in self.thresholds:

            self.reset_stored_z()

            self.trainer.test(
                self.model,
                datamodule=datamodule,
            )

            sparsity = self.measure_sparsity(self.z, threshold)
            results[f"{datamodule_name}_sparsity_{threshold}"] = sparsity.item()

        return results
