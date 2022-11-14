class Task:
    def __init__(self, dataset, metrics):
        return

    def evaluate(self, model, wandb_logger):
        return 1


# Simplest evaluation task, just evaluating standard performance
class StandardEval(Task):
    def __init__(self, dataset, metrics):
        super().__init__(dataset, metrics)

    def evaluate(self, model, wandb_logger):
        # TODO predict on dataset, log standard metrics
        return
