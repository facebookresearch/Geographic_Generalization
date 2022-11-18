from task_utils import Task
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl


class SubsetPerformance(Task):
    def __init__(self, dataset, metrics):
        super().__init__(dataset, metrics)

    def evaluate(self, config, model, trainer):
        # TODO generate predictions and evaluate subset fairness

        return
