import torch
import pytest

from models.beit.beit import BeitBaseClassifierModule, BeitLargeClassifierModule

# from models.clip.clip import (
#     CLIPOPENAI400MClassifierModule,
#     CLIPLAION2BClassifierModule,
#     CLIPLAION400MClassifierModule,
# )
from models.convnext.convnext import (
    ConvNextSmallClassifierModule,
    ConvNextBaseClassifierModule,
    ConvNextLargeClassifierModule,
)

from models.mlp_mixer.mlp_mixer import MLPMixerClassifierModule
from models.regnet.regnet import (
    RegNet2ClassifierModule,
    RegNet4ClassifierModule,
    RegNet6ClassifierModule,
    RegNet8ClassifierModule,
    RegNet16ClassifierModule,
    RegNet32ClassifierModule,
    RegNet64ClassifierModule,
    RegNet120ClassifierModule,
    RegNet320ClassifierModule,
)

from models.resnet.resnet import (
    ResNet18ClassifierModule,
    ResNet50ClassifierModule,
    ResNet101ClassifierModule,
)
from models.rexnet.rexnet import (
    RexNet100ClassifierModule,
    RexNet130ClassifierModule,
    RexNet150ClassifierModule,
    RexNet200ClassifierModule,
)
from models.seer.seer import (
    Seer320ClassifierModule,
    Seer640ClassifierModule,
    Seer1280ClassifierModule,
    Seer10bClassifierModule,
)
from models.tinynet.tinynet import (
    TinyNetAClassifierModule,
    TinyNetBClassifierModule,
    TinyNetCClassifierModule,
    TinyNetDClassifierModule,
    TinyNetEClassifierModule,
)
from models.vgg.vgg import (
    Vgg11ClassifierModule,
    Vgg13ClassifierModule,
    Vgg16ClassifierModule,
    Vgg19ClassifierModule,
)

from models.vit.vit import VitClassifierModule, VitLargeClassifierModule


@pytest.mark.webtest
class TestPreTrainedModels:
    batch_size = 8
    models = [
        # MLPMixerClassifierModule(),
        ResNet18ClassifierModule(),
        # ResNet50ClassifierModule(),
        # ResNet101ClassifierModule(),
        # BeitClassifierModule(),
        VitClassifierModule(),
        # VitLargeClassifierModule(),
        ConvNextSmallClassifierModule(),
        # SimCLRClassifierModule(),
        # CLIPOPENAI400MClassifierModule(),  # only testing 1 CLIP and 1 SEER because they break the testing environment otherwise
        # CLIPLAION400MClassifierModule(),
        # CLIPLAION2BClassifierModule(),
        #  Seer320ClassifierModule(),
        # Seer10bClassifierModule(), this took more than 30min to test independently...
        # Seer640ClassifierModule(),
        # Seer1280ClassifierModule(),
        RegNet2ClassifierModule(),
        # ConvNextSmallClassifierModule(),
        # RexNet100ClassifierModule(),
        TinyNetAClassifierModule(),
        # Vgg11ClassifierModule(),
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
