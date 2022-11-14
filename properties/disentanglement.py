from properties.property_utils import Property


class Base(Property):
    def __init__(self, name):
        super().__init__(name)

    def measure(self, model, datamodule, wandb_logger):
        return 1
