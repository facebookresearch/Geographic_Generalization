from task_utils import Task


class SubsetPerformance(Task):
    def __init__(self, dataset, metrics):
        super().__init__(dataset, metrics)

    def evaluate(self, model, wandb_logger):
        # TODO generate predictions and evaluate subset fairness
        return
