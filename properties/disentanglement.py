from property_utils import Property


class Base(Property):
    def __init__(self):
        super().__init__()

    def measure(model, dataset, wandb_logger):
        return 1
