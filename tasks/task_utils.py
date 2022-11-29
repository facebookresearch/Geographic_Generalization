class Task:
    def __init__(self, dataset, metrics):
        """Task base class defining the structure of a task object.

        Args:
            dataset (str): name of the dataset to instantiate, must be a key in experiment config.
            metrics (list[str]): metrics used for the given task
        """
        self.dataset = dataset
        self.metrics = metrics
        return

    def evaluate(self, config, model, trainer):
        return 1
