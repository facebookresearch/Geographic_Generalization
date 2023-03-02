from pl_bolts.models.self_supervised import SimCLR
import pytorch_lightning as pl
from models.classifier_model import ClassifierModule


class SimCLRClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
        self.feature_dim = 2048

    def load_backbone(self):
        simclr = SimCLR.load_from_checkpoint(self.checkpoint_url, strict=False)
        simclr_encoder = simclr.encoder
        return simclr_encoder

    def forward(self, x):
        return self.model(x)[
            -1
        ]  # https://github.com/Lightning-AI/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/simclr_module.py

    def forward_features(self, x):
        expanded_features = super().forward_features(x)  # batch_size,2048,1,1
        return expanded_features.squeeze(-1).squeeze(-1)  # batch, 2048
