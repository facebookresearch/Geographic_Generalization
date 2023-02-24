import torch
import pytest

from models.resnet.resnet import (
    ResNet18dClassifierModule,
    ResNet50dClassifierModule,
    ResNet101dClassifierModule,
)
from models.mlp_mixer.mlp_mixer import MLPMixerClassifierModule
from models.vit.vit import VitClassifierModule, VitLargeClassifierModule
from models.beit.beit import BeitClassifierModule
from models.simclr.simclr import SimCLRClassifierModule
from models.convnext.convnext import ConvNextlassifierModule


@pytest.mark.webtest
class TestPreTrainedModels:
    batch_size = 8
    models = [
        # MLPMixerClassifierModule(),
        # ResNet18dClassifierModule(),
        # ResNet50dClassifierModule(),
        # ResNet101dClassifierModule(),
        # VitClassifierModule(),
        # BeitClassifierModule(),
        # VitLargeClassifierModule(),
        # ConvNextlassifierModule(),
        SimCLRClassifierModule(),
    ]

    @pytest.mark.parametrize("model", models)
    def test_output_shape(self, model):
        x = torch.rand(self.batch_size, 3, 224, 224)
        z = model(x)
        assert z.shape == (self.batch_size, model.feature_dim)

    @pytest.mark.parametrize("model", models)
    def test_embedding_shape(self, model):
        x = torch.rand(self.batch_size, 3, 224, 224)
        embedding = model.forward_features(x)
        # print(embedding)
        assert len(embedding.shape) == 2
        assert embedding.shape[0] == self.batch_size
