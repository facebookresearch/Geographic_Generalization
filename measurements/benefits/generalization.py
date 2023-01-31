from omegaconf import DictConfig
import pytorch_lightning as pl
from hydra.utils import instantiate
import torchmetrics
import torch.nn.functional as F
from pytorch_lightning.plugins.environments import SLURMEnvironment
import types

from models.classifier_model import ClassifierModule
from measurements.measurement_utils import Measurement


class ClassificationAccuracyEvaluation(Measurement):
    """
    Commented Example of a Measurement (Accuracy on new dataset)

    The base class constructor creates the following objects for you to use:
        1) self.datamodules: a dictionary of datamodules, indexed by the dataset's name (e.g. {dataset_name : datamodule}). This dictionary contains a datamodule for each value passed in the parameter 'dataset_names', so will *only* include datamodules for
            those datasets. To access a datamodule, simply call next(iter(self.datamodules.items()) as seen below to get the dataset name (key) and datamodule (value), or index self.datamodules dictionary with dataset name you want (E.g.: self.datamodules['imagenet']).

        2) self.model: the instantiated model object to use in the measurement.
    """

    def __init__(
        self,
        dataset_names: list[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
    ):
        super().__init__(
            dataset_names=dataset_names,
            model=model,
            experiment_config=experiment_config,
        )

    def measure(
        self,
    ) -> dict[str:float]:
        # 1) Make a dict to store measurements
        results_dict = {}

        # 2) Access datamodules needed - self.datamodules is a dict mapping from dataset names (str) to datamodules. E.g. {'imagenet': ImageNetDatamodule object}
        dataset_name, datamodule = next(iter(self.datamodules.items()))

        # 3) Define the measurement by overwriting the validation step for the model
        new_validation_step = self.make_new_validation_step(dataset_name=dataset_name)
        self.model.validation_step = types.MethodType(new_validation_step, self.model)

        # 4) Use the new validation step to call trainer.validate
        trainer = pl.Trainer(
            **self.experiment_config.trainer,
            plugins=SLURMEnvironment(auto_requeue=False),
        )

        results = trainer.validate(
            model=self.model, datamodule=datamodule
        )  # list[dict]

        # 5) Format results into a dictionary and return
        for d in results:
            results_dict.update(d)

        return results_dict  # to be added to CSV

    def make_new_validation_step(self, dataset_name):
        # The whole purpose of this wrapper function is to allow us to pass in the 'dataset_name' for logging.
        # We cannot pass it into new_validation_step directly without changing the signature required by trainer.

        def new_validation_step(self, batch, batch_idx):
            x, y = batch
            y_hat = self.model(x)
            # If you make a torchmetrics metric outside of the model construction, it doesn't get automatically moved to a device
            metric = torchmetrics.Accuracy().to(self.device)
            result = metric(F.softmax(y_hat, dim=-1), y)
            self.log(f"{dataset_name}_accuracy", result, on_epoch=True)

            return result

        return new_validation_step
