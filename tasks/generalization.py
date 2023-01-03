from tasks.task_utils import Task
from hydra.utils import instantiate
from models.classifier_model import ClassifierModule
from omegaconf import DictConfig
import pytorch_lightning as pl


class StandardEval(Task):
    def __init__(self, dataset_names: list[str], metrics: list[str], logging_name: str):
        """Standard evaluation task, simply calling trainer.validate on the given dataset.

        Args:
            dataset_names (list[str]): list of names of datasets to instantiate, must be a key in experiment config.
            metrics (list[str]): metrics used for the given task.
            logging_name (str): prefix to add to logging for each metric (e.g. 'v2' would log 'v2_val_accuracy')
        """
        super().__init__(dataset_names, metrics, logging_name)

    def evaluate(
        self, config: DictConfig, model: ClassifierModule, trainer: pl.Trainer
    ):
        # Set up a dict to store measurements
        results = {}

        # Make a datamodule
        datamodule_config = getattr(config, self.dataset_names[0])
        datamodule = instantiate(datamodule_config)

        # Perform the Measurement. You can do this a few main ways:
        #      1) manually iterate over datamodule
        #      2) call trainer.validate for basic accuracy measure
        res = trainer.validate(model=model, datamodule=datamodule)[0]
        # first result selected for simplicity

        for metric in res.keys():
            trainer.logger.experiment.log(
                {self.logging_name + "_" + metric: res[metric]}
            )
            results.update({self.logging_name + "_" + metric: res[metric]})

        return results  # to be added to CSV


class AugmentationRobustness(Task):
    def __init__(self, dataset_names: list[str], metrics: list, logging_name: str):
        super().__init__(dataset_names, metrics, logging_name)

    def evaluate(
        self, config: DictConfig, model: ClassifierModule, trainer: pl.Trainer
    ):
        # TODO apply augmentation, evaluate, calculate metrics
        return {self.logging_name: 5}
