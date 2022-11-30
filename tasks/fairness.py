from task_utils import Task
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from models.base_model import BaseModel


class DollarstreetSubsetPerformance(Task):
    def __init__(self, dataset: str, metrics: list[str], logging_prefix: str):
        """Evaluation task for subset performance on Dollarstreet

        Args:
            dataset (str): name of the dataset to instantiate, must be a key in experiment config.
            metrics (list[str]): metrics used for the given task.
            logging_prefix (str): prefix to add to logging
        """
        super().__init__(dataset, metrics, logging_prefix)

    def evaluate(self, config: DictConfig, model: BaseModel, trainer: pl.Trainer):
        # TODO generate predictions and evaluate subset fairness:

        return
