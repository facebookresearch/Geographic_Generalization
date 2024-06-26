"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
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
from typing import Dict, List


class ClassifierModule(pl.LightningModule):
    """
    Base classifier built with Timm.
    """

    def __init__(
        self,
        timm_name: str = "resnet50",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        feature_extraction_layer_index=-2,
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__()
        self.timm_name = timm_name
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.checkpoint_url = checkpoint_url
        self.feature_dim = 1000
        self.linear_eval = linear_eval

        self.model = self.load_model()
        self.feature_extraction_layer_index = feature_extraction_layer_index
        (
            self.backbone,
            self.feature_extraction_layer,
            self.model_layers,
            self.__embedding_dim,
        ) = self.load_backbone()

        self.predictions = pd.DataFrame({})
        self.linear_classifier = torch.nn.Linear(self.embedding_dim, 1000)

        self.train_accuracy = torchmetrics.Accuracy()
        self.val_accuracy = torchmetrics.Accuracy()
        self.test_accuracy = torchmetrics.Accuracy()

    def load_model(self):
        if self.checkpoint_url:
            model = timm.create_model(self.timm_name, pretrained=False)
            print(f"Loading state dict: {self.checkpoint_url}")
            state_dict = torch.load(self.checkpoint_url)["state_dict"]
            new_state_dict = {}
            for k, v in state_dict.items():
                if "model" in k:
                    new_state_dict[k.replace("model.", "")] = v

            model.load_state_dict(new_state_dict)
        else:
            print("Loading Timm model")
            model = timm.create_model(self.timm_name, pretrained=True)
        return model

    def load_backbone(self):
        with torch.no_grad():
            train_nodes, eval_nodes = get_graph_node_names(self.model)
            feature_extraction_layer = eval_nodes[self.feature_extraction_layer_index]
            feature_extractor = create_feature_extractor(
                self.model, return_nodes=[feature_extraction_layer]
            )
            feature_extractor.eval()
            example = torch.rand((1, 3, 224, 224))
            output = feature_extractor(example)[feature_extraction_layer]
            embedding_dim = output.shape[1]
        return feature_extractor, feature_extraction_layer, eval_nodes, embedding_dim

    def forward(self, x):
        if self.linear_eval:
            with torch.no_grad():
                features = self.backbone(x)[self.feature_extraction_layer]
                logits = self.linear_classifier(features)
                return logits
        else:
            return self.model(x)

    def forward_features(self, x):
        return self.backbone(x)[self.feature_extraction_layer]

    @property
    def embedding_dim(self):
        return self.__embedding_dim

    def convert_multilabel_to_single_label(self, y_i):
        split = y_i.split(",")
        if len(split) == 1:
            return int(split[0])
        else:
            return int(split[1])

    def shared_step(self, batch: Tensor, stage: str = "train"):
        # The model expects an image tensor of shape (B, C, H, W)
        x, y, _ = batch
        y_hat = self.model(x)
        y_tensor = torch.LongTensor(
            [self.convert_multilabel_to_single_label(y_i) for y_i in y]
        )
        loss = F.cross_entropy(y_hat, y_tensor)
        accuracy_metric = getattr(self, f"{stage}_accuracy")
        accuracy_metric(F.softmax(y_hat, dim=-1), y_tensor)
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

    def save_predictions(self, predictions):
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

    def on_train_epoch_start(self) -> None:
        if self.linear_eval:
            self.backbone.eval()


if __name__ == "__main__":
    model = ClassifierModule()
