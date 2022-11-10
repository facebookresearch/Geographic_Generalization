
class Task(): 
    def __init__(self):
        return

    def evaluate(model, dataset, metrics):
        return 1

# Simplest evaluation task 
class Base(Task):
    def __init__(self, dataset, metrics):
        super().__init__()
        self.dataset = dataset
        self.metrics = metrics
    
    def evaluate(model,dataset,metrics):
        #TODO predict on dataset, calculate and log subset differences. 
        return
        
# Augmentation Robustness
class AugRobust(Base):
    def __init__(self, dataset, metrics):
        super().__init__()
        self.dataset = dataset
        self.metrics = metrics
    
    def evaluate(model,dataset,metrics):
        #TODO apply augmentation, evaluate, calculate metrics
        return
