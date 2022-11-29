import torch
import pytest

from models.resnet.resnet import ResNet50Pretrained1k, ResNet50Pretrained21k


class TestPreTrainedModels:
    batch_size = 8
    models = [
        ResNet50Pretrained1k(),
        ResNet50Pretrained21k(),
    ]

    @pytest.mark.parametrize("model", models)
    def test_embedding(self, model):
        x = torch.rand(self.batch_size, 3, 224, 224)
        z = model(x)
        assert z.shape == (self.batch_size, model.feature_dim)
