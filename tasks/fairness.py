from generalization import Task, Base

# SubsetFairness
class AugRobust(Base):
    def __init__(self, dataset, metrics):
        super().__init__()
        self.dataset = dataset
        self.metrics = metrics
    
    def evaluate(model,dataset,metrics):
        #TODO generate predictions and 
        return

    