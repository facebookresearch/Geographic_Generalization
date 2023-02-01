import pytest
from hydra import initialize, compose
from hydra.utils import instantiate
import copy
from torch.utils.data import Dataset, DataLoader
from measurements.measurement_utils import Measurement
import pytorch_lightning as pl


class TestMeasurements:
    initialize(version_base=None, config_path="../config/")
    config = compose(config_name="test.yaml")

    model = instantiate(config.model)
    measurement_names = config.measurements
    measurements = []
    for measurement_name in measurement_names:
        measurement_config = getattr(config, measurement_name)
        measurements.append(
            instantiate(
                measurement_config,
                model=copy.deepcopy(model),
                experiment_config=config,
                _recursive_=False,
            )
        )

    @pytest.mark.parametrize("measurement", measurements)
    def test_all_measurements_create_correct_datamodules(
        self, measurement: Measurement
    ):
        assert len(measurement.datamodules) == len(measurement.dataset_names)
        assert list(measurement.datamodules.keys()) == measurement.dataset_names
        datamodule = measurement.datamodules[measurement.dataset_names[0]]
        assert issubclass(type(datamodule), pl.LightningDataModule)
