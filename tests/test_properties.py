from measurements.properties.equivariance.equivariance import Equivariance
import pytest
import torch


class TestEquivariance:
    resnet18_config = {
        "_target_": "models.resnet.resnet.ResNet18dClassifierModule",
        "learning_rate": 1e-4,
        "optimizer": "adam",
    }

    @pytest.fixture(scope="module")
    def equivariance_measure(self):
        equivariance = Equivariance("", dataset_names=["dummy"])
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
        results = equivariance_measure.measure(
            dict(), self.resnet18_config, limit_test_batches=5
        )
        assert equivariance_measure.z.shape == (8 * 5, 512)
        assert equivariance_measure.z_t.shape == (8 * 5, 512, 10)

    def test_results(self, equivariance_measure: Equivariance):
        results = equivariance_measure.measure(
            dict(), self.resnet18_config, limit_test_batches=5
        )

        assert "equivariance_rotate" in results
        assert "invariance_rotate" in results

    def test_shuffle_z_t(self, equivariance_measure: Equivariance):
        z_t = torch.rand(8, 512, 10)
        z_t_shuffled = equivariance_measure.shuffle_z_t(z_t)
        assert z_t.shape == z_t_shuffled.shape
        assert not torch.allclose(z_t, z_t_shuffled)
