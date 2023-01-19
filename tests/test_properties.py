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

    def test_test_step(self):
        pass

    def test_embeddings_are_stored(self, equivariance_measure: Equivariance):
        results = equivariance_measure.measure(
            dict(), self.resnet18_config, limit_test_batches=5
        )
        assert equivariance_measure.z.shape[1] == 512
        assert equivariance_measure.z_t.shape[1] == 512

    def test_results(self, equivariance_measure: Equivariance):
        results = equivariance_measure.measure(
            dict(), self.resnet18_config, limit_test_batches=5
        )

        assert "equivariance_rotate" in results
        assert "invariance_rotate" in results
