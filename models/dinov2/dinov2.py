from models.classifier_model import ClassifierModule
import torch
import torch.nn.functional as F
from torch import Tensor


class DINOv2ViTB14(ClassifierModule):
    def load_model(self):
        classifier = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vitb14_lc", pretrained=True
        )
        return classifier

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch: Tensor, stage: str = "train"):
        # The model expects an image tensor of shape (B, C, H, W)
        x, y = batch
        y_hat = self.model(x)
        loss = F.cross_entropy(y_hat, y)
        accuracy_metric = getattr(self, f"{stage}_accuracy")
        accuracy_metric(F.softmax(y_hat, dim=-1), y)
        self.log(f"{stage}_loss", loss, sync_dist=True)
        self.log(
            f"{stage}_accuracy",
            accuracy_metric,
            prog_bar=True,
            sync_dist=True,
            on_epoch=True,
            on_step=False,
        )

        return loss
