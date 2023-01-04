from omegaconf import DictConfig
from abc import ABC, abstractmethod


class Measurement(ABC):
    """Measure base class defining the structure of a measure object.

    Args:
        logging_name (str): name used in wandb logging (without prefix or suffix)
        model_class (str): path to model object to instantiate
        model_args (dict): dict of args / config to pass into model class to load the model.
        dataset_names (list[str]): list of dataset names
    Return:
        dict in the form {str: float}, where each key represents the name of the measurement, and each float is the corresponding value.
    """

    def __init__(
        self,
        logging_name: str,
        dataset_names: list[str],
    ):
        self.logging_name = logging_name
        self.dataset_names = dataset_names
        return

    @abstractmethod
    def measure(
        self,
        config: DictConfig,
        model_config: dict,
    ):
        return {self.logging_name: 0}
