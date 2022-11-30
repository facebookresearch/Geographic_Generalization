import torch
import pytest

from models.resnet.resnet import (
    ResNet18dClassifierModule,
    ResNet50dClassifierModule,
    ResNet101dClassifierModule,
)


class TestPreTrainedModels:
    batch_size = 8
    models = [
        ResNet18dClassifierModule(),
        ResNet50dClassifierModule(),
        ResNet101dClassifierModule(),
    ]

    @pytest.mark.parametrize("model", models)
    def test_embedding(self, model):
        x = torch.rand(self.batch_size, 3, 224, 224)
        z = model(x)
        assert z.shape == (self.batch_size, model.feature_dim)
