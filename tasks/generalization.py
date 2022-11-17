from task_utils import Task, StandardEval

# Augmentation Based Robustness
class AugRobust(Task):
    def __init__(self, dataset, metrics):
        super().__init__(dataset, metrics)

    def evaluate(self, model, trainer):
        # TODO apply augmentation, evaluate, calculate metrics

        return
