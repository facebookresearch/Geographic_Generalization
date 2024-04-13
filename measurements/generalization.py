"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from omegaconf import DictConfig
import torchmetrics
import torch.nn.functional as F
from models.classifier_model import ClassifierModule
from measurements.measurement_utils import Measurement
import types

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
        accuracies = self.model.predictions[["id", f"accurate_top{n}"]]
        incomes_and_regions = self.datamodules[datamodule_name].file["test"][
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


class ClassificationAccuracyEvaluation(Measurement):
    """
    Commented Example of a Measurement (Accuracy on new datamodule)

    The base class constructor creates the following objects for you to use:
        1) self.datamodules: a dictionary of datamodules, indexed by the datamodule's name (e.g. {datamdoule_name : datamodule}). This dictionary contains a datamodule for each value passed in the parameter 'datamodule_names', so will *only* include datamodules for
            those datamodules. To access a datamodule, simply call next(iter(self.datamodules.items()) as seen below to get the datamodule name (key) and datamodule (value), or index self.datamodules dictionary with datamodule name you want (E.g.: self.datamodules['imagenet']).

        2) self.model: the instantiated model object to use in the measurement.

    The base class also has a function 'save_extra_results_to_csv' which can be used to save model predictions, or other details.
    """

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
        # 1) Make a dict to store measurements
        results_dict = {}

        for data_module_name, datamodule in self.datamodules.items():
            print(data_module_name)

            # 2) Access datamodules needed - self.datamodules is a dict mapping from datamodule names (str) to datamodules. E.g. {'imagenet': ImageNetDatamodule object}
            datamodule_name, datamodule = next(iter(self.datamodules.items()))

            # 3) Define the measurement by overwriting the test step for the model
            new_test_step = self.make_new_test_step(
                datamodule_name=datamodule_name, mask=datamodule.mask
            )

            self.model.test_step = types.MethodType(new_test_step, self.model)
            print("made new test step")

            # Call validate / test function
            results = self.trainer.test(model=self.model, datamodule=datamodule)

            # 5) Format results into a dictionary and return
            for d in results:
                results_dict.update(d)

            # Optional: save predictions
            if self.save_detailed_results:
                self.save_extra_results_to_csv(
                    extra_results=self.model.predictions,
                    name=f"{datamodule_name}_predictions",
                )

        return results_dict  # to be added to CSV

    def make_new_test_step(self, datamodule_name, mask):
        # The whole purpose of this wrapper function is to allow us to pass in the 'datamodule_name' for logging.
        # We cannot pass it into new_test_step directly without changing the signature required by trainer.

        def new_test_step(self, batch, batch_idx):
            x, y = batch
            if mask is not None:
                y_hat = self(x)[:, mask]
            else:
                y_hat = self(x)
            # If you make a torchmetrics metric outside of the model construction, it doesn't get automatically moved to a device

            result = self.test_accuracy(F.softmax(y_hat, dim=-1), y)
            self.log(f"{datamodule_name}_test_accuracy", result, on_epoch=True)

            self.save_predictions(
                {"prediction": y_hat.cpu().tolist(), "label": y.cpu().tolist()}
            )

            return result

        return new_test_step
