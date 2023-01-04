from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra.utils import instantiate
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning.plugins.environments import SLURMEnvironment
import types

from measurements.measurement_utils import Measurement


class BasicEvaluation(Measurement):
    """Example measure of a measurement"""

    def __init__(self, logging_name: str, dataset_names: list[str]):
        super().__init__(logging_name=logging_name, dataset_names=dataset_names)

    def measure(
        self,
        config: DictConfig,
        model_config: dict,
    ):
        # Make a dict to store measurements
        results_dict = {}

        # Make your model
        model = instantiate(model_config)

        # Make a datamodule (config contains all datasets in config/dataset_library, so you can just reference by the name!)
        first_dataset_name = self.dataset_names[0]
        datamodule_config = getattr(config, first_dataset_name)
        datamodule = instantiate(datamodule_config)

        # Perform the Measurement! There are a few paradigms you can use, shown as options below.

        # - Option 1: Lightning version (overwrite @test_step function in model)
        new_validation_step = self.make_new_validation_step(self.logging_name)

        model.validation_step = types.MethodType(new_validation_step, model)

        trainer = pl.Trainer(
            **config.trainer,
            plugins=SLURMEnvironment(auto_requeue=False),
        )

        results = trainer.validate(model=model, datamodule=datamodule)  # list[dict]

        for d in results:
            results_dict.update(d)

        # - Option 2: Manual (get datataloader from datamodule, change how you want, measure)

        # test_loader = datamodule.test_dataloader()

        # for batch, id in enumerate(test_loader):
        #    new_metric = 1  # calculate something
        #    pass

        # results_dict = {self.logging_name: new_metric}

        return results_dict  # to be added to CSV

    def make_new_validation_step(self, logging_name):
        # The whole purpose of this wrapper function is to allow us to pass in the property's 'logging_name'.
        # We cannot pass it into new_validation_step directly without changing the signature required by trainer.

        def new_validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            # If you make a torchmetrics metric outside of the model construction, it doesn't get automatically moved to a device
            metric = torchmetrics.Accuracy().to(self.device)
            result = metric(F.softmax(y_hat, dim=-1), y)
            self.log(logging_name + "_accuracy", result, on_epoch=True)

            return result

        return new_validation_step
