from task_utils import Task
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from models.classifier_model import ClassifierModule


class DollarstreetSubsetPerformance(Task):
    def __init__(self, dataset_names: list[str], metrics: list[str], logging_name: str):
        """Evaluation task for subset performance on Dollarstreet

        Args:
            dataset (list[str]): names of datasets to instantiate, each must be a key in experiment config.
            metrics (list[str]): metrics used for the given task.
            logging_name (str): prefix to add to logging (e.g. a prefix of 'v2' would become 'v2_val_accuracy')
        """
        super().__init__(dataset_names, metrics, logging_name)

    def evaluate(
        self, config: DictConfig, model: ClassifierModule, trainer: pl.Trainer
    ):
        # TODO generate predictions and evaluate subset fairness:

        return {self.logging_name: 50}
