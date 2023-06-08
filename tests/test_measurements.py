import pytest
from hydra import initialize, compose
from hydra.utils import instantiate
import copy
from measurements.measurement_utils import Measurement
import pytorch_lightning as pl
import pytest
import torch
import hydra
from hydra import initialize, compose
from hydra.utils import instantiate
import numpy as np


@pytest.mark.webtest
class TestMeasurementSetUp:
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
        assert len(measurement.datamodules) == len(measurement.datamodule_names)
        assert list(measurement.datamodules.keys()) == measurement.datamodule_names
        datamodule = measurement.datamodules[measurement.datamodule_names[0]]
        assert issubclass(type(datamodule), pl.LightningDataModule)
        hydra.core.global_hydra.GlobalHydra.instance().clear()


class TestGeneralization:
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../config/")
    config = compose(config_name="test.yaml")

    model = instantiate(config.model)
    measurement_names = config.measurements
    generalization_measurements = [
        x for x in measurement_names if "generalization" in x
    ]

    measurements = []
    for measurement_name in generalization_measurements:
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
    def test_generalization_measurements_report_correct_values(
        self, measurement: Measurement
    ):
        result = measurement.measure()
        assert len(result) > 0
        for val in list(result.values()):
            assert val is not None
            assert val > 0.0
        hydra.core.global_hydra.GlobalHydra.instance().clear()
