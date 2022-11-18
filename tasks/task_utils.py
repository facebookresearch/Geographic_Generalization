from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl
from config_utils import find_config_object


class Task:
    def __init__(self, dataset, metrics):
        self.dataset = dataset
        self.metrics = metrics
        return

    def evaluate(self, config, model, trainer):
        return 1


# Simplest evaluation task, just evaluating standard performance
class StandardEval(Task):
    def __init__(self, dataset, metrics):
        super().__init__(dataset, metrics)

    def evaluate(self, config, model, trainer):
        datamodule_config = getattr(config, self.dataset)
        datamodule = instantiate(datamodule_config)
        trainer.validate(model=model, datamodule=datamodule)

        return
