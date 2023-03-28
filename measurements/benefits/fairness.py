from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
import types
import torch
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
        accuracies = self.model.predictions[["id", "accurate_top5"]]
        incomes_and_regions = self.datamodules["dollarstreet"].file[
            ["id", "Income_Group", "region"]
        ]
        combined = pd.merge(accuracies, incomes_and_regions, on="id", how="left")

        avg_acc_by_income = (
            combined.groupby("Income_Group")["accurate_top5"].mean().to_dict()
        )
        avg_acc_by_region = combined.groupby("region")["accurate_top5"].mean().to_dict()

        return avg_acc_by_income, avg_acc_by_region

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
            datamodule_name=datamodule_name, mask=datamodule.mask
        )

        self.model.test_step = types.MethodType(new_test_step, self.model)

        # Calculate overall results and add to results dictionary
        results = self.trainer.test(model=self.model, datamodule=datamodule)
        for d in results:
            results_dict.update(d)

        self.save_extra_results_to_csv(
            extra_results=self.model.predictions, name="dollarstreet_results"
        )

        # Calculate disparities and add to results dictionary
        acc_by_income, acc_by_region = self.calculate_disparities()

        acc_by_income = {
            f"dollarstreet-{k.lower()}_test_accuracy": v
            for k, v in acc_by_income.items()
        }
        acc_by_region = {
            f"dollarstreet-{k.lower()}_test_accuracy": v
            for k, v in acc_by_region.items()
        }

        results_dict.update(acc_by_region)
        results_dict.update(acc_by_income)

        # Save extra results to CSVs
        if self.save_detailed_results == "True":

            acc_by_income = self.convert_float_dict_to_list_dict(acc_by_income)
            acc_by_region = self.convert_float_dict_to_list_dict(acc_by_region)

            self.save_extra_results_to_csv(
                extra_results=acc_by_income,
                name=f"{datamodule_name}_accuracy_by_income",
            )
            self.save_extra_results_to_csv(
                extra_results=acc_by_region,
                name=f"{datamodule_name}_accuracy_by_region",
            )

            self.save_extra_results_to_csv(
                extra_results=self.model.predictions,
                name=f"{datamodule_name}_predictions",
            )

        return results_dict

    def make_new_test_step(self, datamodule_name, mask):
        def new_test_step(self, batch, batch_idx):
            x, y, identifier = batch

            y_hat = self(x)  # [:, mask]

            confidences5, indices5 = torch.nn.functional.softmax(y_hat, dim=-1).topk(5)
            confidences1, indices1 = torch.nn.functional.softmax(y_hat, dim=-1).topk(1)

            acc5s = []
            acc1s = []
            for i in range(len(y)):
                y_int = [int(x) for x in y[i].split(",")]
                acc5 = len(set(y_int) & set(indices5[i].tolist())) > 0
                acc1 = len(set(y_int) & set(indices1[i].tolist())) > 0
                acc5s.append(acc5)
                acc1s.append(acc1)

            self.log(datamodule_name + "_test_accuracy", np.mean(acc5s), on_epoch=True)

            self.save_predictions(
                {
                    "id": list(identifier),
                    "output": y_hat.cpu().tolist(),
                    "predictions": indices5.cpu().tolist(),
                    "confidences": confidences5.cpu().tolist(),
                    "label": list(y),
                    "accurate_top1": acc1s,
                    "accurate_top5": acc5s,
                }
            )

            return

        return new_test_step
