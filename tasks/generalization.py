from tasks.task_utils import Task
from hydra.utils import instantiate

# Simplest evaluation task, just evaluating standard performance
class StandardEval(Task):
    def __init__(self, dataset, metrics):
        """Standard evaluation task, simply calling trainer.validate on the given dataset.

        Args:
            dataset (str): name of the dataset to instantiate, must be a key in experiment config.
            metrics (list[str]): metrics used for the given task.
        """
        super().__init__(dataset, metrics)

    def evaluate(self, config, model, trainer):
        datamodule_config = getattr(config, self.dataset)
        datamodule = instantiate(datamodule_config)
        trainer.validate(model=model, datamodule=datamodule)
        return


# Augmentation Based Robustness
class AugRobust(Task):
    def __init__(self, dataset, metrics):
        super().__init__(dataset, metrics)

    def evaluate(self, config, model, trainer):
        # TODO apply augmentation, evaluate, calculate metrics
        return
