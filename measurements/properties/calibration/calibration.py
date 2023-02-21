from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
import torch
from measurements.properties.equivariance import transformations
from torch import Tensor
from typing import List, Dict
from models.classifier_model import ClassifierModule
import torch.nn.functional as F


class NLL(Measurement):
    """Negative Log-Likelihood"""

    def __init__(self,
        datamodule_names: List[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
    ):
        super().__init__(datamodule_names, model, experiment_config)
        self.model.test_step = self.test_step

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        return loss

    def measure(self):
        results = dict()

        for datamodule_name, datamodule in self.datamodules.items():
            results[f"{datamodule_name}_nll"] = self.trainer.test(
                self.model,
                datamodule=datamodule,
            )

        return results
