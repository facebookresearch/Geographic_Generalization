from tasks.task_utils import Task
from hydra.utils import instantiate
from models.base_model import BaseModel
from omegaconf import DictConfig
import pytorch_lightning as pl


class StandardEval(Task):
    def __init__(self, dataset: BaseModel, metrics: list, logging_prefix: str):
        """Standard evaluation task, simply calling trainer.validate on the given dataset.

        Args:
            dataset (str): name of the dataset to instantiate, must be a key in experiment config.
            metrics (list[str]): metrics used for the given task.
            logging_prefix (str): prefix to add to logging for each metric (e.g. 'v2' would log 'v2_val_accuracy')
        """
        super().__init__(dataset, metrics, logging_prefix)

    def evaluate(self, config: DictConfig, model: BaseModel, trainer: pl.Trainer):
        datamodule_config = getattr(config, self.dataset)
        datamodule = instantiate(datamodule_config)
        res = trainer.validate(model=model, datamodule=datamodule)[0]
        for metric in res.keys():
            trainer.logger.experiment.log(
                {self.logging_prefix + "_" + metric: res[metric]}
            )

        return


class AugRobust(Task):
    def __init__(self, dataset: BaseModel, metrics: list, logging_prefix: str):
        super().__init__(dataset, metrics, logging_prefix)

    def evaluate(self, config: DictConfig, model: BaseModel, trainer: pl.Trainer):
        # TODO apply augmentation, evaluate, calculate metrics
        return
