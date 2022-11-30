from models.base_model import BaseModel
from omegaconf import DictConfig
import pytorch_lightning as pl


class Task:
    def __init__(self, dataset: str, metrics: list[str], logging_prefix: str):
        """Task base class defining the structure of a task object.

        Args:
            dataset (str): name of the dataset to instantiate, must be a key in experiment config.
            metrics (list[str]): metrics used for the given task
            logging_prefix (str): string to add to logging metric
        """
        self.dataset = dataset
        self.metrics = metrics
        self.logging_prefix = logging_prefix
        return

    def evaluate(self, config: DictConfig, model: BaseModel, trainer: pl.Trainer):
        return 1
