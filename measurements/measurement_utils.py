from omegaconf import DictConfig
from abc import ABC, abstractmethod
from models.classifier_model import ClassifierModule
from hydra.utils import instantiate
from datasets.image_datamodule import ImageDataModule


class Measurement(ABC):
    """Measure base class defining the structure of a measure object.

    This base class constructor creates the following objects for use:
    1) self.datamodules: a dictionary of datamodules, indexed by the datamodule's name (e.g. {datamodule_name : datamodule}). This dictionary contains a datamodule for each value passed in the parameter 'datamodule_names', so will *only* include datamodules for
        those datamodules. To access a datamodule, simply index the self.datamodules dictionary with datamodule name you want (again, this key must be a value in datamodule_names parameter). Example: self.datamodules['imagenet'].

    2) self.model: the instantiated model object to use in the measurement.

    Args:
        datamodule_names (list[str]): list of datamodule names required for this measurement. E.g. ['imagenet', 'dollarstreet']
        model (ClassifierModule): pytorch model to perform the measurement with
        experiment_config (DictConfig): Hydra config used primarily to instantiate a trainer. Must have key: 'trainer' to be compatible with pytorch lightning.
    Return:
        dict in the form {str: float}, where each key represents the name of the measurement, and each float is the corresponding value. Keys should be in the form: <datamodule_name>_<property_name>.
    """

    def __init__(
        self,
        datamodule_names: list[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
    ):
        self.datamodules = self.make_datamodules(experiment_config, datamodule_names)
        self.experiment_config = experiment_config
        self.model = model
        self.datamodule_names = datamodule_names
        return

    def make_datamodules(
        self, experiment_config: DictConfig, datamodule_names: list[str]
    ) -> dict[str:ImageDataModule]:
        datamodules = {}
        for datamodule_name in datamodule_names:
            datamodule_config = getattr(experiment_config, datamodule_name)
            datamodule = instantiate(datamodule_config)
            datamodules[datamodule_name] = datamodule

        return datamodules

    @abstractmethod
    def measure(
        self,
    ) -> dict[str:float]:
        datamodule_name, datamodule = next(iter(self.datamodules.items()))
        property_name = "property"

        return {f"{datamodule_name}_{property_name}": 0.1}
