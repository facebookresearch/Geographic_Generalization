from task_utils import Task, StandardEval

# Augmentation Robustness
class AugRobust(Task):
    def __init__(self, dataset, metrics):
        super().__init__(dataset, metrics)

    def evaluate(self, model, wandb_logger):
        # TODO apply augmentation, evaluate, calculate metrics
        return
