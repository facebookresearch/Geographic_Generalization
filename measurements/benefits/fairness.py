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

    def calculate_disparities(self, datamodule_name="dollarstreet", n=5):
        print(self.datamodules[datamodule_name].file.columns)
        accuracies = self.model.predictions[["id", f"accurate_top{n}"]]
        incomes_and_regions = self.datamodules[datamodule_name].file[
            ["id", "Income_Group", "Region"]
        ]
        combined = pd.merge(accuracies, incomes_and_regions, on="id", how="left")

        avg_acc_by_income = (
            combined.groupby("Income_Group")[f"accurate_top{n}"].mean().to_dict()
        )
        avg_acc_by_region = (
            combined.groupby("Region")[f"accurate_top{n}"].mean().to_dict()
        )

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
        acc_by_income1, acc_by_region1 = self.calculate_disparities(
            datamodule_name=datamodule_name, n=1
        )
        acc_by_income5, acc_by_region5 = self.calculate_disparities(
            datamodule_name=datamodule_name, n=5
        )

        acc_by_income1 = {
            f"dollarstreet-{k.lower()}_test_accuracy1": v
            for k, v in acc_by_income1.items()
        }
        acc_by_income5 = {
            f"dollarstreet-{k.lower()}_test_accuracy1": v
            for k, v in acc_by_income5.items()
        }
        acc_by_region1 = {
            f"dollarstreet-{k.lower()}_test_accuracy1": v
            for k, v in acc_by_region1.items()
        }
        acc_by_region5 = {
            f"dollarstreet-{k.lower()}_test_accuracy5": v
            for k, v in acc_by_region5.items()
        }

        results_dict.update(acc_by_region1)
        results_dict.update(acc_by_region5)
        results_dict.update(acc_by_income1)
        results_dict.update(acc_by_income5)

        # Save extra results to CSVs
        if self.save_detailed_results == "True":
            acc_by_income1 = self.convert_float_dict_to_list_dict(acc_by_income1)
            acc_by_region1 = self.convert_float_dict_to_list_dict(acc_by_region1)
            acc_by_income5 = self.convert_float_dict_to_list_dict(acc_by_income5)
            acc_by_region5 = self.convert_float_dict_to_list_dict(acc_by_region5)

            self.save_extra_results_to_csv(
                extra_results=acc_by_income1,
                name=f"{datamodule_name}_accuracy_by_income1",
            )
            self.save_extra_results_to_csv(
                extra_results=acc_by_income5,
                name=f"{datamodule_name}_accuracy_by_income5",
            )
            self.save_extra_results_to_csv(
                extra_results=acc_by_region1,
                name=f"{datamodule_name}_accuracy_by_region1",
            )
            self.save_extra_results_to_csv(
                extra_results=acc_by_region5,
                name=f"{datamodule_name}_accuracy_by_region5",
            )

            self.save_extra_results_to_csv(
                extra_results=self.model.predictions,
                name=f"{datamodule_name}_predictions",
            )

        return results_dict

    def make_new_test_step(self, datamodule_name, mask):
        def new_test_step(self, batch, batch_idx):
            x, y, identifier = batch

            y_hat = self(x)

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

            self.log(datamodule_name + "_test_accuracy5", np.mean(acc5s), on_epoch=True)
            self.log(datamodule_name + "_test_accuracy1", np.mean(acc1s), on_epoch=True)

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


class GeodePerformance(Measurement):
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

    def calculate_disparities(self, datamodule_name="geode", n=1):
        print(self.model.predictions.columns)
        accuracies = self.model.predictions[["id", f"accurate_top{n}"]]

        incomes_and_regions = self.datamodules[datamodule_name].file[["id", "region"]]
        combined = pd.merge(accuracies, incomes_and_regions, on="id", how="left")

        # Check that the merge happened successfully
        assert len(combined) == len(accuracies)
        assert combined.isna().sum().sum() == 0

        avg_acc_by_region = (
            combined.groupby("region")[f"accurate_top{n}"].mean().to_dict()
        )

        return avg_acc_by_region

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
            mask=datamodule.mask,
            label_col=datamodule.label_col,
        )

        self.model.test_step = types.MethodType(new_test_step, self.model)

        # Calculate overall results and add to results dictionary
        results = self.trainer.test(model=self.model, datamodule=datamodule)
        for d in results:
            results_dict.update(d)

        self.save_extra_results_to_csv(
            extra_results=self.model.predictions, name="geode_results"
        )

        # Calculate disparities and add to results dictionary
        acc_by_region5 = self.calculate_disparities(
            datamodule_name=datamodule_name, n=5
        )
        acc_by_region1 = self.calculate_disparities(
            datamodule_name=datamodule_name, n=1
        )

        acc_by_region5 = {
            f"geode-{k.lower()}_test_accuracy5": v for k, v in acc_by_region5.items()
        }
        acc_by_region1 = {
            f"geode-{k.lower()}_test_accuracy1": v for k, v in acc_by_region1.items()
        }

        results_dict.update(acc_by_region5)
        results_dict.update(acc_by_region1)

        # Save extra results to CSVs

        print("saving detailed results")
        acc_by_region5 = self.convert_float_dict_to_list_dict(acc_by_region5)
        acc_by_region1 = self.convert_float_dict_to_list_dict(acc_by_region1)

        self.save_extra_results_to_csv(
            extra_results=acc_by_region5,
            name=f"{datamodule_name}_accuracy_by_region5",
        )
        self.save_extra_results_to_csv(
            extra_results=acc_by_region1,
            name=f"{datamodule_name}_accuracy_by_region1",
        )

        self.save_extra_results_to_csv(
            extra_results=self.model.predictions,
            name=f"{datamodule_name}_predictions",
        )

        return results_dict

    def make_new_test_step(
        self,
        datamodule_name,
        mask,
        label_col,
    ):
        def new_test_step(self, batch, batch_idx):
            x, y, identifier = batch

            y_hat = self(x)

            confidences5, indices5 = torch.nn.functional.softmax(y_hat, dim=-1).topk(5)
            confidences1, indices1 = torch.nn.functional.softmax(y_hat, dim=-1).topk(1)

            acc5s = []
            acc1s = []

            for i in range(len(y)):
                if "1k" in label_col:
                    y_int = [int(x) for x in y[i].split(",")]
                    acc5 = len(set(y_int) & set(indices5[i].tolist())) > 0
                    acc1 = len(set(y_int) & set(indices1[i].tolist())) > 0
                else:
                    y_int = [int(y[i].item())]
                    acc5 = len(set(y_int) & set(indices5[i].tolist())) > 0
                    acc1 = len(set(y_int) & set(indices1[i].tolist())) > 0

                acc5s.append(acc5)
                acc1s.append(acc1)

            self.log(datamodule_name + "_test_accuracy1", np.mean(acc1s), on_epoch=True)
            self.log(datamodule_name + "_test_accuracy5", np.mean(acc5s), on_epoch=True)

            self.save_predictions(
                {
                    "id": identifier,
                    "output": y_hat.cpu().tolist(),
                    "predictions": indices5.cpu().tolist(),
                    "confidences": confidences5.cpu().tolist(),
                    "label": list(y)
                    if (type(y) == list or y.device == "cpu")
                    else list(y.cpu()),
                    "accurate_top1": acc1s,
                    "accurate_top5": acc5s,
                }
            )

            return

        return new_test_step
