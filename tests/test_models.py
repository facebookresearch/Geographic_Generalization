import torch
import pytest

from models.resnet.resnet import (
    ResNet18ClassifierModule,
    ResNet50ClassifierModule,
    ResNet101ClassifierModule,
)
from models.mlp_mixer.mlp_mixer import MLPMixerClassifierModule
from models.vit.vit import VitClassifierModule, VitLargeClassifierModule
from models.beit.beit import BeitClassifierModule
from models.simclr.simclr import SimCLRClassifierModule
from models.convnext.convnext import ConvNextlassifierModule
from models.clip.clip import (
    CLIPOPENAI400MClassifierModule,
    CLIPLAION2BClassifierModule,
    CLIPLAION400MClassifierModule,
)
from models.seer.seer import (
    Seer320ClassifierModule,
    Seer640ClassifierModule,
    Seer1280ClassifierModule,
    Seer10bClassifierModule,
)


@pytest.mark.webtest
class TestPreTrainedModels:
    batch_size = 8
    models = [
        MLPMixerClassifierModule(),
        ResNet18ClassifierModule(),
        ResNet50ClassifierModule(),
        ResNet101ClassifierModule(),
        BeitClassifierModule(),
        VitClassifierModule(),
        VitLargeClassifierModule(),
        ConvNextlassifierModule(),
        SimCLRClassifierModule(),
        CLIPOPENAI400MClassifierModule(),  # only testing 1 CLIP and 1 SEER because they break the testing environment otherwise
        # CLIPLAION400MClassifierModule(),
        # CLIPLAION2BClassifierModule(),
        #  Seer320ClassifierModule(),
        # Seer10bClassifierModule(), this took more than 30min to test independently...
        # Seer640ClassifierModule(),
        # Seer1280ClassifierModule(),
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
