from omegaconf import DictConfig
import torchmetrics
import torch.nn.functional as F
from models.classifier_model import ClassifierModule
from measurements.measurement_utils import Measurement
import types


class ClassificationAccuracyEvaluation(Measurement):
    """
    Commented Example of a Measurement (Accuracy on new datamodule)

    The base class constructor creates the following objects for you to use:
        1) self.datamodules: a dictionary of datamodules, indexed by the datamodule's name (e.g. {datamdoule_name : datamodule}). This dictionary contains a datamodule for each value passed in the parameter 'datamodule_names', so will *only* include datamodules for
            those datamodules. To access a datamodule, simply call next(iter(self.datamodules.items()) as seen below to get the datamodule name (key) and datamodule (value), or index self.datamodules dictionary with datamodule name you want (E.g.: self.datamodules['imagenet']).

        2) self.model: the instantiated model object to use in the measurement.

    The base class also has a function 'save_extra_results_to_csv' which can be used to save model predictions, or other details.
    """

    def __init__(
        self,
        datamodule_names: list[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
    ):
        super().__init__(
            datamodule_names=datamodule_names,
            model=model,
            experiment_config=experiment_config,
        )

    def measure(
        self,
    ) -> dict[str:float]:
        # 1) Make a dict to store measurements
        results_dict = {}

        for data_module_name, datamodule in self.datamodules.items():

            # 2) Access datamodules needed - self.datamodules is a dict mapping from datamodule names (str) to datamodules. E.g. {'imagenet': ImageNetDatamodule object}
            datamodule_name, datamodule = next(iter(self.datamodules.items()))

            # 3) Define the measurement by overwriting the test step for the model
            new_test_step = self.make_new_test_step(
                datamodule_name=datamodule_name, mask=datamodule.mask
            )
            self.model.test_step = types.MethodType(new_test_step, self.model)

            # Call validate / test function
            results = self.trainer.test(model=self.model, datamodule=datamodule)

            # 5) Format results into a dictionary and return
            for d in results:
                results_dict.update(d)

            # Optional: save predictions
            self.save_extra_results_to_csv(
                extra_results=self.model.predictions,
                name=f"{datamodule_name}_predictions",
            )

        return results_dict  # to be added to CSV

    def make_new_test_step(self, datamodule_name, mask):
        # The whole purpose of this wrapper function is to allow us to pass in the 'datamodule_name' for logging.
        # We cannot pass it into new_test_step directly without changing the signature required by trainer.

        def new_test_step(self, batch, batch_idx):
            x, y = batch
            if mask is not None:
                y_hat = self.model(x)[:, mask]
            else:
                y_hat = self.model(x)
            # If you make a torchmetrics metric outside of the model construction, it doesn't get automatically moved to a device
            metric = torchmetrics.Accuracy().to(self.device)
            result = metric(F.softmax(y_hat, dim=-1), y)
            self.log(f"{datamodule_name}_test_accuracy", result, on_epoch=True)

            self.save_predictions(
                {"prediction": y_hat.cpu().tolist(), "label": y.cpu().tolist()}
            )

            return result

        return new_test_step
