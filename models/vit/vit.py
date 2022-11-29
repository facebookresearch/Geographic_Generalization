import pytorch_lightning as pl
import timm


class ViTPretrained21k(pl.LightningModule):
    """ViT with Aug Reg pretrained on ImageNet 21k"""

    def __init__(
        self,
    ):
        super().__init__()

        self.model = self.load_backbone()
        self.feature_dim = 768

    def load_backbone(self):
        # trained on ImageNet-21k with aug reg
        # https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py#L82
        vit = timm.create_model("vit_base_patch16_224", pretrained=True)
        return vit

    def forward(self, x):
        feats = self.model.forward_features(x)
        return feats


class ViTPretrained1k(ViTPretrained21k):
    """ViT pretrained on ImageNet-1k"""

    def load_backbone(self):
        # https://github.com/Alibaba-MIIL/ImageNet21K
        vit = timm.create_model("vit_base_patch16_224_miil", pretrained=True)
        return vit
