from pl_bolts.models.self_supervised import SimCLR
import pytorch_lightning as pl
from typing import Optional, Tuple
import torch


class SimCLRPretrained(pl.LightningModule):
    """Loads a pretrained SimCLR model"""

    def __init__(
        self,
    ):
        super().__init__()

        self.feature_dim = 2048
        self.model = self.load_backbone()

    def load_backbone(self):
        weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
        simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)
        return simclr

    def forward(self, x):
        return self.model(x)
