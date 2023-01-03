from models.base_model import BaseModel
from omegaconf import DictConfig
import pytorch_lightning as pl
from abc import ABC, abstractmethod


class Task(ABC):
    def __init__(self, dataset_names: list[str], metrics: list[str], logging_name: str):
        """Task base class defining the structure of a task object.

        Args:
            dataset_names (list[str]): name of the dataset to instantiate, must be a key in experiment config.
            metrics (list[str]): metrics used for the given task
            logging_name (str): string to add to logging metric (e.g. a prefix of 'v2' would become 'v2_val_accuracy')
        """
        self.dataset_names = dataset_names
        self.metrics = metrics
        self.logging_name = logging_name
        return
    
    @abstractmethod
    def evaluate(self, config: DictConfig, model: BaseModel, trainer: pl.Trainer):
        return 1
