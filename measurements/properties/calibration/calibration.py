from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
from typing import List, Dict
from models.classifier_model import ClassifierModule
import torch.nn.functional as F
import types


class NLL(Measurement):
    """Negative Log-Likelihood"""

    def __init__(self,
        datamodule_names: List[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
    ):
        super().__init__(datamodule_names, model, experiment_config)

    def make_new_test_step(self, datamodule_name):
        def test_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            loss = F.cross_entropy(y_hat, y)
            self.log(f"{datamodule_name}_nll", loss, on_epoch=True)
            return loss

        return test_step

    def measure(self):
        results_dict = dict()

        for datamodule_name, datamodule in self.datamodules.items():

            new_test_step = self.make_new_test_step(datamodule_name)
            # self.model.test_step = new_test_step
            self.model.test_step = types.MethodType(new_test_step, self.model)
            results = self.trainer.test(
                self.model,
                datamodule=datamodule,
            )
            for d in results:
                results_dict.update(d)

        return results_dict
