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


class DollarStreetPerformance(Measurement):
    """Example measure of a measurement"""

    def __init__(self, logging_name: str, dataset_names: list[str]):
        super().__init__(logging_name=logging_name, dataset_names=dataset_names)

    def measure(
        self,
        config: DictConfig,
        model_config: dict,
    ) -> dict[str:float]:

        results_dict = {}

        # Make your model
        model = instantiate(model_config)

        # Make a datamodule
        first_dataset_name = self.dataset_names[0]
        datamodule_config = getattr(config, first_dataset_name)
        datamodule = instantiate(datamodule_config)

        new_validation_step = self.make_new_validation_step(
            self.logging_name, self.convert_predictions_to_imagenet1k_labels
        )

        model.validation_step = types.MethodType(new_validation_step, model)

        trainer = pl.Trainer(
            **config.trainer,
            plugins=SLURMEnvironment(auto_requeue=False),
        )

        results = trainer.validate(model=model, datamodule=datamodule)  # list[dict]

        for d in results:
            results_dict.update(d)

        return results_dict  # to be added to CSV

    def convert_predictions_to_imagenet1k_labels(self, pred_indices):
        names = []
        for pred in pred_indices.numpy():
            pred_names = [imagenet_classes.IMAGENET1K_IDX_TO_NAMES[idx] for idx in pred]
            names.append(pred_names)
        return names

    def make_new_validation_step(self, logging_name, pred_conversion):
        def new_validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)

            confidences, indices = torch.nn.functional.softmax(y_hat).topk(5)
            preds = pred_conversion(indices.cpu())

            accs = []
            for i in range(len(y)):
                all_preds = set(sum([x.split(", ") for x in preds[i]], []))
                acc = len(all_preds & set(y[i].split(", "))) > 0
                accs.append(acc)

            self.log(logging_name + "_accuracy", np.mean(accs), on_epoch=True)

            return 1

        return new_validation_step
