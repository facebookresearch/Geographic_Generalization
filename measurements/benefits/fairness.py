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
import pandas as pd


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

    def calculate_disparities(self):
        accuracies = self.model.predictions[["url", "accurate_top5"]]
        incomes_and_geographies = self.datamodules["dollarstreet"].file[
            ["url", "income_bucket", "region"]
        ]
        combined = pd.merge(accuracies, incomes_and_geographies, on="url", how="left")

        avg_acc_by_region = combined.groupby("region")["accurate_top5"].mean().to_dict()
        avg_acc_by_income = (
            combined.groupby("income_bucket")["accurate_top5"].mean().to_dict()
        )

        return avg_acc_by_region, avg_acc_by_income

    def convert_float_dict_to_list_dict(self, d: dict):
        for k in d.keys():
            d[k] = [d[k]]
        return d

    def measure(
        self,
    ) -> dict[str:float]:

        results_dict = {}

        datamodule_name, datamodule = next(iter(self.datamodules.items()))

        new_test_step = self.make_new_test_step(
            datamodule_name=datamodule_name,
            pred_conversion=self.convert_predictions_to_imagenet1k_labels,
        )

        self.model.test_step = types.MethodType(new_test_step, self.model)

        # Calculate overall results and add to results dictionary
        results = self.trainer.test(model=self.model, datamodule=datamodule)
        for d in results:
            results_dict.update(d)

        # Calculate disparities and add to results dictionary
        acc_by_region, acc_by_income = self.calculate_disparities()

        acc_by_region = {
            "dollarstreet_test_accuracy_region_" + k: v
            for k, v in acc_by_region.items()
        }
        acc_by_income = {
            "dollarstreet_test_accuracy_income_" + k: v
            for k, v in acc_by_income.items()
        }

        results_dict.update(acc_by_region)
        results_dict.update(acc_by_income)

        # Save extra results to CSVs
        if self.save_detailed_results == "True":
            acc_by_region = self.convert_float_dict_to_list_dict(acc_by_region)
            acc_by_income = self.convert_float_dict_to_list_dict(acc_by_income)

            self.save_extra_results_to_csv(
                extra_results=acc_by_region,
                name=f"{datamodule_name}_accuracy_by_region",
            )
            self.save_extra_results_to_csv(
                extra_results=acc_by_income,
                name=f"{datamodule_name}_accuracy_by_income",
            )

            self.save_extra_results_to_csv(
                extra_results=self.model.predictions,
                name=f"{datamodule_name}_predictions",
            )

        return results_dict

    def convert_predictions_to_imagenet1k_labels(self, pred_indices):
        names = []
        for pred in pred_indices.numpy():
            pred_names = [imagenet_classes.IMAGENET1K_IDX_TO_NAMES[idx] for idx in pred]
            names.append(pred_names)
        return names

    def make_new_test_step(self, datamodule_name, pred_conversion):
        def new_test_step(self, batch, batch_idx):
            x, y, url = batch
            y_hat = self.model(x)

            confidences5, indices5 = torch.nn.functional.softmax(y_hat, dim=-1).topk(5)
            confidences1, indices1 = torch.nn.functional.softmax(y_hat, dim=-1).topk(1)
            preds5 = pred_conversion(indices5.cpu())
            preds1 = pred_conversion(indices1.cpu())

            acc5s = []
            acc1s = []
            for i in range(len(y)):
                all_preds5 = set(sum([x.split(", ") for x in preds5[i]], []))
                all_preds1 = set(sum([x.split(", ") for x in preds1[i]], []))
                acc5 = len(all_preds5 & set(y[i].split(", "))) > 0
                acc1 = len(all_preds1 & set(y[i].split(", "))) > 0
                acc5s.append(acc5)
                acc1s.append(acc1)

            self.log(datamodule_name + "_test_accuracy", np.mean(acc5s), on_epoch=True)

            self.save_predictions(
                {
                    "url": list(url),
                    "output": y_hat.cpu().tolist(),
                    "predictions": preds5,
                    "confidences": confidences5.cpu().tolist(),
                    "label": list(y),
                    "accurate_top1": acc1s,
                    "accurate_top5": acc5s,
                }
            )

            return 1

        return new_test_step
