from omegaconf import DictConfig
from abc import ABC, abstractmethod


class Measurement(ABC):
    """Measure base class defining the structure of a measure object.

    Args:
        logging_name (str): common prefix to use for all logged metrics in the measurement. E.g. 'imagenet_v2'
        dataset_names (list[str]): list of dataset names required for this measurement. E.g. ['imagenet', 'dollarstreet']
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
    ) -> dict[str:float]:
        return {self.logging_name: 0}
