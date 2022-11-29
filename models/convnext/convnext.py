import pytorch_lightning as pl
from timm import create_model


class ConvNextPretrained1k(pl.LightningModule):
    """ConvNext based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py"""

    def __init__(self):
        super().__init__()
        self.feature_dim = 1024
        self.model_name = "convnext_base_in22k"
        self.model = self.load_backbone()

    def load_backbone(self):
        """loads ConvNext model"""
        model = create_model(self.model_name, pretrained=True, num_classes=0)
        return model

    def forward(self, x):
        return self.model.forward(x)


class ConvNextPretrained21k(ConvNextPretrained1k):
    def __init__(self):
        super().__init__()
        self.feature_dim = 1024
        self.model_name = "convnext_base"
        self.model = self.load_backbone()
