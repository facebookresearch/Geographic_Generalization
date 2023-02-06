from measurements.properties.equivariance.equivariance import Equivariance
import pytest
import torch

from hydra import initialize, compose
from hydra.utils import instantiate


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
