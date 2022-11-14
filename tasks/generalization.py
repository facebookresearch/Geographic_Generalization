from task_utils import Task
        
# Augmentation Robustness
class AugRobust(Task):
    def __init__(self, dataset, metrics):
        super().__init__()
        self.dataset = dataset
        self.metrics = metrics
    
    def evaluate(model):
        #TODO apply augmentation, evaluate, calculate metrics
        return
