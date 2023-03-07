from measurements.measurement_utils import Measurement
from omegaconf import DictConfig
from typing import List
from models.classifier_model import ClassifierModule
import torch
import torch.nn.functional as F
import types
import numpy as np


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
            x, y = batch[0], batch[1]
            y_hat = self.model(x)
            if isinstance(y, list) and isinstance(y[0], str):  # multi-label case
                y_hat = y_hat.cpu()
                y_new = np.zeros(len(y), dtype=int)
                # TODO: not sure if this conversion maybe
                # should happen in the dataloader?
                y = [np.array(sorted([int(x) for x in y[i].split(',')]), dtype=int) for i in range(len(y))]
                for i in range(len(y)):
                    y_new[i] = y[i][np.argmax(y_hat[i][y[i]])]
                y = y_new
                y, y_hat = torch.from_numpy(y).to(x.device), y_hat.to(x.device)
            loss = F.cross_entropy(y_hat, y)
            self.log(f"{datamodule_name}_calibration_nll", loss, on_epoch=True)
            return loss

        return test_step

    def measure(self):
        results_dict = dict()

        for datamodule_name, datamodule in self.datamodules.items():

            new_test_step = self.make_new_test_step(datamodule_name)
            self.model.test_step = types.MethodType(new_test_step, self.model)
            results = self.trainer.test(
                self.model,
                datamodule=datamodule,
            )
            for d in results:
                results_dict.update(d)

        return results_dict


class ECE(Measurement):
    """Expected Calibration Error"""

    def __init__(self,
        datamodule_names: List[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
        n_bins: int = 15,
    ):
        super().__init__(datamodule_names, model, experiment_config)
        self.n_bins = n_bins

    def make_new_test_step(self):
        def test_step(self, batch, batch_idx):
            x, y = batch[0], batch[1]
            if isinstance(y, torch.Tensor):
                y = y.cpu().tolist()
            if isinstance(y, list) and isinstance(y[0], str):  # multi-label case
                # TODO: not sure if this conversion maybe
                # should happen in the dataloader?
                y = [[int(x) for x in y[i].split(',')] for i in range(len(y))]
            y_hat = self.model(x)
            self.save_predictions(
                {"prediction": F.softmax(y_hat, dim=-1).cpu().tolist(), "label": y}
            )
            return None

        return test_step

    @staticmethod
    def measure_ece(preds, targets, n_bins=15):
        """ Adapted from https://github.com/SamsungLabs/pytorch-ensembles/blob/master/metrics.py
        Args:
            preds: numpy array of shape (num samples, num classes)
            targets: numpy array of shape (num samples,)
            n_bins: number of bins in [0, 1] range for ECE estimation
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
        if isinstance(targets[0], list) or isinstance(targets[0], np.ndarray):  # multi-label case
            accuracies = np.zeros(len(targets))
            for i in range(len(targets)):
                accuracies[i] = predictions[i] in targets[i]
        else:
            accuracies = (predictions == targets)

        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)
            if prop_in_bin > 0:
                accuracy_in_bin = np.mean(accuracies[in_bin])
                avg_confidence_in_bin = np.mean(confidences[in_bin])
                delta = avg_confidence_in_bin - accuracy_in_bin
                ece += np.abs(delta) * prop_in_bin
        return ece

    def measure(self):
        results_dict = dict()

        for datamodule_name, datamodule in self.datamodules.items():

            new_test_step = self.make_new_test_step()
            self.model.test_step = types.MethodType(new_test_step, self.model)
            self.trainer.test(
                self.model,
                datamodule=datamodule,
            )

            ece_val = self.measure_ece(
                np.array(self.model.predictions["prediction"].tolist()),
                np.array(self.model.predictions["label"].tolist()),
                n_bins=self.n_bins
            )
            results_dict[f"{datamodule_name}_calibration_ece"] = ece_val

        return results_dict
