from measurements.properties.equivariance.equivariance import Equivariance


class TestEquivariance:
    resnet18_config = {
        "_target_": "models.resnet.resnet.ResNet18dClassifierModule",
        "learning_rate": 1e-4,
        "optimizer": "adam",
    }

    def test_embeddings_are_stored(self):
        equivariance = Equivariance("", dataset_names=["dummy"])
        results = equivariance.measure(dict(), self.resnet18_config)
        assert "equivariance_rotate" in results
