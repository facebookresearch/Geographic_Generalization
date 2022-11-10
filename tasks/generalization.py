
class Task(): 
    __init__(config: DictConfig):
    self.config = config

    def evaluate(model, dataset, metrics):
        return 1


class Base(Task):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def evaluate(model,dataset,metrics):
        print("started generalization task")
        print(dataset)
        print(metrics)
        # predict on dataset, calculate and log subset differences. 

