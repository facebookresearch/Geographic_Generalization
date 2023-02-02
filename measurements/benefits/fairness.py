from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra.utils import instantiate
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning.plugins.environments import SLURMEnvironment
import types
import torch
from datasets import imagenet_classes
import numpy as np
from models.classifier_model import ClassifierModule


class DollarStreetPerformance(Measurement):
    def __init__(
        self,
        datamodule_names: list[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
    ):
        super().__init__(
            datamodule_names=datamodule_names,
            model=model,
            experiment_config=experiment_config,
        )

    def measure(
        self,
    ) -> dict[str:float]:

        results_dict = {}

        datamodule_name, datamodule = next(iter(self.datamodules.items()))

        new_validation_step = self.make_new_validation_step(
            datamodule_name=datamodule_name,
            pred_conversion=self.convert_predictions_to_imagenet1k_labels,
        )

        self.model.validation_step = types.MethodType(new_validation_step, self.model)

        results = self.trainer.validate(
            model=self.model, datamodule=datamodule
        )  # list[dict]

        for d in results:
            results_dict.update(d)

        return results_dict

    def convert_predictions_to_imagenet1k_labels(self, pred_indices):
        names = []
        for pred in pred_indices.numpy():
            pred_names = [imagenet_classes.IMAGENET1K_IDX_TO_NAMES[idx] for idx in pred]
            names.append(pred_names)
        return names

    def make_new_validation_step(self, datamodule_name, pred_conversion):
        def new_validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)

            confidences, indices = torch.nn.functional.softmax(y_hat, dim=-1).topk(5)
            preds = pred_conversion(indices.cpu())

            accs = []
            for i in range(len(y)):
                all_preds = set(sum([x.split(", ") for x in preds[i]], []))
                acc = len(all_preds & set(y[i].split(", "))) > 0
                accs.append(acc)

            self.log(datamodule_name + "_test_accuracy", np.mean(accs), on_epoch=True)

            return 1

        return new_validation_step
