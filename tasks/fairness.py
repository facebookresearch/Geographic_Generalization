from task_utils import Task
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl


class DollarstreetSubsetPerformance(Task):
    def __init__(self, dataset, metrics):
        """Evaluation task for subset performance on Dollarstreet

        Args:
            dataset (str): name of the dataset to instantiate, must be a key in experiment config.
            metrics (list[str]): metrics used for the given task.
        """
        super().__init__(dataset, metrics)

    def evaluate(self, config, model, trainer):
        # TODO generate predictions and evaluate subset fairness

        return
