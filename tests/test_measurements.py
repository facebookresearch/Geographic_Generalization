import pytest
from hydra import initialize, compose
from hydra.utils import instantiate
import copy
from measurements.measurement_utils import Measurement
import pytorch_lightning as pl
from measurements.properties.equivariance.equivariance import Equivariance
from measurements.properties.sparsity.sparsity import Sparsity
import pytest
import torch
import hydra
from hydra import initialize, compose
from hydra.utils import instantiate


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


@pytest.mark.webtest
class TestEquivariance:
    @pytest.fixture(scope="module")
    def equivariance_measure(self):
        initialize(version_base=None, config_path="../config/")
        experiment_config = compose(config_name="test.yaml")
        model = instantiate(experiment_config.model)
        equivariance = Equivariance(["dummy"], model, experiment_config)
        return equivariance

    def test_test_step(self, equivariance_measure: Equivariance):
        batch_size = 8
        batch = (
            torch.rand(batch_size, 3, 224, 224),
            torch.randint(10, (batch_size, 1)),
        )
        equivariance_measure.reset_stored_z()
        equivariance_measure.test_step(batch, 0)
        assert equivariance_measure.z.shape == (
            batch_size,
            512,
        )
        # embedding dim x number of transformation parameters
        assert equivariance_measure.z_t.shape == (batch_size, 512, 10)

    def test_embeddings_are_stored(self, equivariance_measure: Equivariance):
        equivariance_measure.reset_stored_z()
        equivariance_measure.measure()
        num_batches = equivariance_measure.experiment_config.trainer.limit_test_batches
        assert equivariance_measure.z.shape == (num_batches * 8, 512)
        assert equivariance_measure.z_t.shape == (num_batches * 8, 512, 10)

    def test_results(self, equivariance_measure: Equivariance):
        equivariance_measure.reset_stored_z()
        results = equivariance_measure.measure()

        assert "dummy_equivariance_rotate" in results
        assert results["dummy_equivariance_rotate"] > 0.0
        assert "dummy_invariance_rotate" in results
        assert results["dummy_invariance_rotate"] > 0.0

    def test_shuffle_z_t(self, equivariance_measure: Equivariance):
        z_t = torch.rand(8, 512, 10)
        z_t_shuffled = equivariance_measure.shuffle_z_t(z_t)
        assert z_t.shape == z_t_shuffled.shape
        assert not torch.allclose(z_t, z_t_shuffled)


class TestSparsity:
    @pytest.fixture(scope="module")
    def sparsity_measure(self):
        hydra.core.global_hydra.GlobalHydra.instance().clear()
        initialize(version_base=None, config_path="../config/")
        experiment_config = compose(config_name="test.yaml")
        model = instantiate(experiment_config.model)
        sparsity = Sparsity(["v2"], model, experiment_config)
        return sparsity

    def test_test_step(self, sparsity_measure: Sparsity):
        batch_size = 8
        batch = (
            torch.rand(batch_size, 3, 224, 224),
            torch.randint(10, (batch_size, 1)),
        )
        sparsity_measure.reset_stored_z()
        sparsity_measure.test_step(batch, 0)
        assert sparsity_measure.z.shape == (
            batch_size,
            512,
        )

    def test_embeddings_are_stored(self, sparsity_measure: Sparsity):
        sparsity_measure.reset_stored_z()
        sparsity_measure.measure()
        num_batches = sparsity_measure.experiment_config.trainer.limit_test_batches
        assert sparsity_measure.z.shape == (num_batches * 32, 512)

    def test_results(self, sparsity_measure: Sparsity):
        sparsity_measure.reset_stored_z()
        results = sparsity_measure.measure()

        assert len(results) > 0
        assert "v2_sparsity_1" in results
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
