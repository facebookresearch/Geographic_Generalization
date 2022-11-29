from models.base_model import ShapeNetBaseModel
from typing import Tuple, Optional
import pytorch_lightning as pl
import torch
import timm


class MLPMixerPretrained1k(pl.LightningModule):
    """Pretrained MLP Mixer model"""

    def __init__(
        self,
    ):
        super().__init__()
        self.model = self.load_backbone()
        self.feature_dim = 768

    def load_backbone(self):
        # trained on ImageNet1k
        mlp_mixer = timm.create_model("mixer_b16_224", pretrained=True)
        return mlp_mixer

    def forward(self, x):
        feats = self.model.forward_features(x)
        return feats


class MLPMixerPretrained21k(MLPMixerPretrained1k):
    def load_backbone(self):
        # trained on ImageNet-21k
        # https://github.com/Alibaba-MIIL/ImageNet21K
        model = timm.create_model('mixer_b16_224_miil_in21k', pretrained=True)
        return model
