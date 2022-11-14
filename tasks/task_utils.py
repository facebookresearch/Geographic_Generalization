class Task(): 
    def __init__(self, dataset, metrics):
        return

    def evaluate(model):
        return 1

# Simplest evaluation task, just evaluating standard performance
class Standard_Eval(Task):
    def __init__(self, dataset, metrics):
        super().__init__()
        self.dataset = dataset
        self.metrics = metrics
    
    def evaluate(model):
        #TODO predict on dataset, log standard metrics
        return