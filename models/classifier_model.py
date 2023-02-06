import os
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Any
import torchmetrics
import torch.nn.functional as F
import timm
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
import pandas as pd


class ClassifierModule(pl.LightningModule):
    """
    Base classifier built with Timm.
    """

    def __init__(
        self,
        timm_name: str = "resnet50d",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        feature_extraction_layer_index=-2,
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth",
    ):
        super().__init__()
        self.timm_name = timm_name
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.checkpoint_url = checkpoint_url
        self.feature_dim = 1000

        self.model = self.load_backbone()
        self.feature_extraction_layer_index = feature_extraction_layer_index
        (
            self.feature_extractor,
            self.feature_extraction_layer,
            self.model_layers,
        ) = self.load_feature_extractor()

        self.predictions = pd.DataFrame({})

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def load_backbone(self):
        model = timm.create_model(self.timm_name, pretrained=True)
        state_dict = torch.utils.model_zoo.load_url(self.checkpoint_url)
        model.load_state_dict(state_dict)
        return model

    def load_feature_extractor(self):
        train_nodes, eval_nodes = get_graph_node_names(self.model)
        feature_extraction_layer = eval_nodes[self.feature_extraction_layer_index]
        feature_extractor = create_feature_extractor(
            self.model, return_nodes=[feature_extraction_layer]
        )
        return feature_extractor, feature_extraction_layer, eval_nodes

    def forward(self, x):
        return self.model(x)

    def forward_features(self, x):
        return self.feature_extractor(x)[self.feature_extraction_layer]

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

    def save_predictions(self, predictions: dict[str:list]):
        preds = pd.DataFrame(predictions)
        self.predictions = pd.concat([self.predictions, preds])
        return

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


if __name__ == "__main__":
    model = ClassifierModule()
