from properties.property_utils import Property


class Aug_Approximation(Property):
    """Example measure of an equivariance property to be completed"""

    def __init__(self):
        super().__init__()
        self.name = "equiv_aug"

    def measure(self, model, datamodule, trainer):
        trainer.logger.experiment.log({self.name: 13})
        return 1
