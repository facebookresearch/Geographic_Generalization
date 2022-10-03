from models import evaluation
from models.clip.clip import CLIPPretrained
import pytorch_lightning as pl
import torch
from datasets.dummy import ShapeNetDataModule
from models.convnext.convnext import ConvNextPretrained1k, ConvNextPretrained21k
from models.ibot.ibot import iBotPretrained1k, iBotPretrained21k
from models.mae.mae import MAEPretrained
import pytest

from models.clip.clip import CLIPZeroShotPretrained

from models.mlp_mixer.mlp_mixer import MLPMixerPretrained1k, MLPMixerPretrained21k
from models.resnet.resnet import ResNet50Pretrained1k, ResNet50Pretrained21k
from models.simclr.simclr import SimCLRPretrained
from models.vicreg.vicreg import VICRegPretrained1k, VICRegPretrained21k
from models.vit.vit import ViTPretrained1k, ViTPretrained21k


class DummyBackbone(pl.LightningModule):
    def __init__(self, size=224):
        super().__init__()
        self.feature_dim = size
        self.linear = torch.nn.Linear(size, size)

    def forward(self, x):
        # forward on one channel
        out = self.linear(x[:, 0, 0, :])
        return out


class TestEvaluation:
    batch_size = 8
    num_classes = 50
    backbone = DummyBackbone()
    dm = ShapeNetDataModule(num_classes=num_classes)

    def test_finetuning(self):
        finetuner = evaluation.Finetuning(self.backbone, datamodule=self.dm)
        x = torch.rand(self.batch_size, 3, 224, 224)
        z = self.backbone(x)
        assert z.shape == (self.batch_size, 224)

        y_hat = finetuner(x)
        assert y_hat.shape == (self.batch_size, self.num_classes)

    def test_linear_eval(self):
        linear_eval = evaluation.LinearEval(self.backbone, datamodule=self.dm)
        x = torch.rand(self.batch_size, 3, 224, 224)
        z = self.backbone(x)
        assert z.shape == (self.batch_size, 224)

        y_hat = linear_eval(x)
        assert y_hat.shape == (self.batch_size, self.num_classes)

    def test_store_predictions(self):
        backbone = ViTPretrained1k()
        ckpt_dir = "/checkpoint/marksibrahim/logs/robustness-limits/vit_pretrained_1k_shapenet_random_class_split/2022-09-02_14-20-36/58"
        store_predictions = evaluation.StorePredictions(backbone, ckpt_dir, self.dm)
        assert hasattr(store_predictions, "evaluator")


class TestPreTrainedModels:
    batch_size = 8
    models = [
        CLIPPretrained(),
        MAEPretrained(),
        MLPMixerPretrained1k(),
        MLPMixerPretrained21k(),
        ResNet50Pretrained1k(),
        ResNet50Pretrained21k(),
        SimCLRPretrained(),
        ViTPretrained21k(),
        ViTPretrained1k(),
        iBotPretrained1k(),
        iBotPretrained21k(),
        ConvNextPretrained1k(),
        ConvNextPretrained21k(),
        VICRegPretrained1k(),
        VICRegPretrained21k(),
    ]

    @pytest.mark.parametrize("model", models)
    def test_embedding(self, model):
        x = torch.rand(self.batch_size, 3, 224, 224)
        z = model(x)
        assert z.shape == (self.batch_size, model.feature_dim)


class TestZeroShot:
    def test_clip_forward(self):
        clip = CLIPZeroShotPretrained()
        x = torch.rand(4, 3, 224, 224)
        probs = clip(x)
        assert probs.shape == (4, 54)
