class Property (): 
    __init__(config: DictConfig):
    self.config = config

    def measure(model, datasets, config):
        return

# Class defines each property, configs define which measures of that property to use
class Disentanglement(Property):
    def __init__(self, config):
        super().__init__()
        metrics = config.metrics # example: ['DCI']
    
    def measure(model: BaseModel, datasets: list, wandb_logger: WandbLogger):
        for metric in self.metrics: 
            m = locals()[metric](model, datasets)
            wandb_logger.log({metric: m})

    # define functions for each metric 
    def DCI(model: BaseModel, datasets: list):
        return 1

# Class defines each property, configs define which measures of that property to use
class Equivariance(Property):
    def __init__(self, config):
        super().__init__()
        metrics = config.metrics # example: ['approximate_equivariance']
    
    def measure(model: BaseModel, datasets: list, wandb_logger: WandbLogger):
        for metric in self.metrics: 
            m = locals()[metric](model, datasets)
            wandb_logger.log({metric: m})

    # define functions for each metric 
    def approximate_equivariance(model: BaseModel, datasets: list):
        return 1

