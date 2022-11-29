from properties.property_utils import Property


class DCI(Property):
    """Example measure of a disentanglement to be completed"""

    def __init__(self):
        super().__init__()
        self.name = "dci"

    def measure(self, model, datamodule, trainer):
        trainer.logger.experiment.log({self.name: 5})
        return 1
