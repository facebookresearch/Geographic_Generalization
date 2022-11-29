import torch
from torch import Tensor
import pytorch_lightning as pl
import torch.nn.functional as F
import timm
import torchmetrics


class ResNet50Pretrained1k(pl.LightningModule):
    """Resnet 50 with recipe from https://arxiv.org/abs/2110.00476"""

    def __init__(self):
        super().__init__()
        self.feature_dim = 2048
        self.model = self.load_backbone()

    def load_backbone(self):
        """Loads ResNet-50 weights from ResNet Strikes back paper.
        Weights based on the A1 training recipe aiming for the best performance
        with 600 epochs.
        https://arxiv.org/abs/2110.00476
        Weights from https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-rsb-weights
        """
        checkpoint_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth"
        model = timm.create_model("resnet50d")
        state_dict = torch.utils.model_zoo.load_url(checkpoint_url)
        model.load_state_dict(state_dict)

        model = torch.nn.Sequential(
            *(list(model.children())[:-1] + []),
        )
        return model

    def forward(self, x):
        return self.model(x)


class ResNet50Pretrained21k(ResNet50Pretrained1k):
    def load_backbone(self):
        """Loads ResNet pretrained on 21k from https://github.com/Alibaba-MIIL/ImageNet21K"""
        # ignore classifier head
        model = timm.create_model(
            "tresnet_m_miil_in21k", pretrained=True, num_classes=0
        )
        return model

    def forward(self, x):
        return self.model(x)


class ResNet50ClassifierModule(pl.LightningModule):
    """Confirmed this yields 79.3% Top-1 Val Accuracy"""

    def __init__(self, learning_rate: float = 1e-4, optimizer: str = "adam"):
        super().__init__()
        self.learning_rate = learning_rate
        self.optimizer = optimizer

        self.model = self.load_backbone()

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def load_backbone(self):
        """Loads ResNet-50 weights from ResNet Strikes back paper.
        Weights based on the A1 training recipe aiming for the best performance
        with 600 epochs.
        https://arxiv.org/abs/2110.00476
        Weights from https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-rsb-weights
        """
        checkpoint_url = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth"
        model = timm.create_model("resnet50d")
        state_dict = torch.utils.model_zoo.load_url(checkpoint_url)
        model.load_state_dict(state_dict)
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
