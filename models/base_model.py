import os
import pytorch_lightning as pl
import torch
from torch import Tensor
from typing import Dict, Any
import torchmetrics
import torch.nn.functional as F


class BaseModel(pl.LightningModule):
    """PyTorch Lightning wrapper used to define basic metrics logging and checkpointing."""

    def __init__(self):
        super().__init__()
        self.save_hyperparameters(ignore=["datamodule", "backbone"])
        # note metrics are automatically tracked in self.trainer.logged_metrics
        # self.results stores additional results
        self.results: Dict[str, Any] = dict()

    @property
    def model_name(self):
        return self.__class__.__name__

    @property
    def logged_metrics(self) -> Dict[str, Any]:
        """Converts all logged metrics to json serializable pure Python values"""
        metrics = {"model_name": self.model_name}
        if not self.trainer:
            return metrics
        for name, value in self.trainer.logged_metrics.items():
            if isinstance(value, torch.Tensor):
                try:
                    value = value.item()
                except ValueError:
                    value = value.tolist()
            metrics[name] = value
        return metrics

    def on_save_checkpoint(self, checkpoint):
        checkpoint["seed"] = int(os.getenv("PL_GLOBAL_SEED", default=0))
        checkpoint["current_epoch"] = self.current_epoch
        checkpoint["datamodule_hparams"] = self.datamodule_hparams
        if "metrics" not in checkpoint:
            checkpoint["metrics"] = {}
        for metric in self.trainer.logged_metrics:
            checkpoint["metrics"][metric] = self.trainer.logged_metrics[metric]

    @property
    def datamodule_hparams(self) -> dict:
        if not hasattr(self, "datamodule"):
            return None
        return self.datamodule.hparams

    def on_load_checkpoint(self, checkpoint):
        self.seed = checkpoint["seed"]
        if "metrics" not in checkpoint:
            checkpoint["metrics"] = {}
        for metric in checkpoint["metrics"]:
            value = checkpoint["metrics"][metric]
            print(metric, value)

    def _setup_loader_names(self):
        if self.datamodule:
            self.train_loader_names = self.datamodule.train_loader_names
            self.val_loader_names = self.datamodule.val_loader_names
            self.test_loader_names = self.datamodule.test_loader_names
            if self.num_classes is None:
                self.num_classes = self.datamodule.num_classes
        else:
            print("loader names not loaded from datamodule")
            if self.num_classes is None:
                print("assuming dataset contains 15 classes")
                self.num_classes = 15
            self.train_loader_names = []
            self.val_loader_names = []
            self.test_loader_names = []

    def setup_accuracy_metrics(self):
        loader_types = (
            self.train_loader_names + self.val_loader_names + self.test_loader_names
        )

        for k in self.top_k:
            for data_type in loader_types:
                setattr(
                    self,
                    f"{data_type}_top_{k}_accuracy",
                    torchmetrics.Accuracy(top_k=k),
                )

    def setup_per_class_accuracy(self):
        loader_types = (
            self.train_loader_names + self.val_loader_names + self.test_loader_names
        )

        print(f"Tracking per class accuracy for {self.num_classes} classes")

        for k in self.top_k:
            for data_type in loader_types:
                labels = [f"{data_type}_{c}" for c in range(self.num_classes)]
                setattr(
                    self,
                    f"{data_type}_top_{k}_per_class_accuracy",
                    torchmetrics.wrappers.ClasswiseWrapper(
                        torchmetrics.Accuracy(
                            num_classes=self.num_classes, average=None
                        ),
                        labels=labels,
                    ),
                )


class ShapeNetBaseModel(BaseModel):
    """Module for training on ShapeNet.

    Inherit and implement:
        - load_backbone()
        - forward()
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.top_k = kwargs["top_k"]
        # num_classes is set within _setup_loader_names()
        self.num_classes = kwargs.get("num_classes", None)
        self.datamodule = kwargs["datamodule"]

        self._setup_loader_names()
        if self.top_k:
            self.setup_accuracy_metrics()

    def load_backbone(self):
        raise NotImplementedError("model needs to implement a backbone")

    def forward(self):
        raise NotImplementedError("model needs to implement a forward method")

    def shared_step(self, batch: Tensor, stage: str = "train"):
        x, y, _ = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        batch_size = x.shape[0]
        self.log(
            f"{stage}_loss",
            loss,
            sync_dist=True,
            # loader names are used instead
            add_dataloader_idx=False,
            batch_size=batch_size,
        )
        for k in self.top_k:
            accuracy_metric = getattr(self, f"{stage}_top_{k}_accuracy")
            accuracy_metric(F.softmax(y_hat, dim=-1), y)
            self.log(
                f"{stage}_top_{k}_accuracy",
                accuracy_metric,
                prog_bar=True,
                sync_dist=True,
                on_epoch=True,
                on_step=False,
                batch_size=batch_size,
                # loader names are used instead
                add_dataloader_idx=False,
            )
        return loss

    def training_step(self, loaders, loader_idx):
        loss = 0
        for loader_name in loaders:
            batch = loaders[loader_name]
            loss += self.shared_step(batch, stage=loader_name)
        return loss

    def validation_step(self, batch, batch_idx, loader_idx=0):
        loader_name = self.val_loader_names[loader_idx]
        loss = self.shared_step(batch, stage=loader_name)
        return loss

    def test_step(self, batch, batch_idx, loader_idx=0):
        loader_name = self.test_loader_names[loader_idx]
        loss = self.shared_step(batch, stage=loader_name)
        return loss

    def configure_optimizers(self):
        if self.optimizer == "adam":
            return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer == "sgd":
            return self.sgd()
        raise ValueError(f"optimizer {self.optimizer} not implemented")

    def sgd(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.learning_rate,
            momentum=self.momentum,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]
