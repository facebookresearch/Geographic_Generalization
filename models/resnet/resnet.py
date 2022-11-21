import torch
from torch import Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
import timm
import torchmetrics


class ClassifierModule(pl.LightningModule):
    """
    Base classifier built with Timm.
    """

    def __init__(
        self,
        timm_name: str = "resnet50d",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth",
    ):
        super().__init__()
        self.timm_name = timm_name
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.checkpoint_url = checkpoint_url

        self.model = self.load_backbone()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def load_backbone(self):
        model = timm.create_model(self.timm_name, pretrained=True)
        state_dict = torch.utils.model_zoo.load_url(self.checkpoint_url)
        model.load_state_dict(state_dict)
        # print(f"Model created with weights from {self.checkpoint_url}")
        return model

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

    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="train")
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="val")
        return loss

    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch, stage="test")
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class ResNet50dClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "resnet50d",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth",
        **kwargs,
    ):
        super(ResNet50dClassifierModule, self).__init__(
            timm_name=timm_name,
            checkpoint_url=checkpoint_url,
            **kwargs,
        )


class ResNet101dClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "resnet101d",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth",
        **kwargs,
    ):
        super(ResNet101dClassifierModule, self).__init__(
            timm_name=timm_name,
            checkpoint_url=checkpoint_url,
            **kwargs,
        )


class ResNet18ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "resnet18d",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth",
        **kwargs,
    ):
        super(ResNet18ClassifierModule, self).__init__(
            timm_name=timm_name,
            checkpoint_url=checkpoint_url,
            **kwargs,
        )
