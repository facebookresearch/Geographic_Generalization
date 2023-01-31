from omegaconf import DictConfig
from abc import ABC, abstractmethod
from models.classifier_model import ClassifierModule
from hydra.utils import instantiate
from datasets.image_datamodule import ImageDataModule


class Measurement(ABC):
    """Measure base class defining the structure of a measure object.

    This base class constructor creates the following objects for use:
    1) self.datamodules: a dictionary of datamodules, indexed by the dataset's name (e.g. {dataset_name : datamodule}). This dictionary contains a datamodule for each value passed in the parameter 'dataset_names', so will *only* include datamodules for
        those datasets. To access a datamodule, simply index the self.datamodules dictionary with dataset name you want (again, this key must be a value in dataset_names parameter). Example: self.datamodules['imagenet'].

    2) self.model: the instantiated model object to use in the measurement.

    Args:
        dataset_names (list[str]): list of dataset names required for this measurement. E.g. ['imagenet', 'dollarstreet']
        model (ClassifierModule): pytorch model to perform the measurement with
        experiment_config (DictConfig): Hydra config used primarily to instantiate a trainer. Must have key: 'trainer' to be compatible with pytorch lightning.
    Return:
        dict in the form {str: float}, where each key represents the name of the measurement, and each float is the corresponding value. Keys should be in the form: <dataset_name>_<property_name>.
    """

    def __init__(
        self,
        dataset_names: list[str],
        model: ClassifierModule,
        experiment_config: DictConfig,
    ):
        self.datamodules = self.make_datamodules(experiment_config, dataset_names)
        self.experiment_config = experiment_config
        self.model = model
        self.dataset_names = dataset_names
        return

    def make_datamodules(
        self, experiment_config: DictConfig, dataset_names: list[str]
    ) -> dict[str:ImageDataModule]:
        datamodules = {}
        for dataset_name in dataset_names:
            datamodule_config = getattr(experiment_config, dataset_name)
            datamodule = instantiate(datamodule_config)
            datamodules[dataset_name] = datamodule

        return datamodules

    @abstractmethod
    def measure(
        self,
    ) -> dict[str:float]:
        dataset_name, datamodule = next(iter(self.datamodules.items()))
        property_name = "property"

        return {f"{dataset_name}_{property_name}": 0.1}
