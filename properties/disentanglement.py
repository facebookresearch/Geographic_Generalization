class Property (): 
    __init__(config: DictConfig):
    self.config = config

    def measure(model, datasets, config):
        return

# Class defines each property metric
class DCI(Property):
    def __init__(self, config):
        super().__init__()
    
    def measure(model: BaseModel, datasets: list, wandb_logger: WandbLogger):
        # Calculate
        m = 1
        wandb_logger.log({'DCI': m})



