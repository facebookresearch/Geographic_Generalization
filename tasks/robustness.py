
class OOD_Robustness(Task):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def evaluate(model,config,logger):
        for datasets in config.datasets: 
            #model.evaluate()
        
    