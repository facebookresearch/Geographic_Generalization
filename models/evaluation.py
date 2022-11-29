from models.base_model import BaseModel
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero
from typing import Optional, Tuple
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from models.loggers import find_existing_checkpoint
import os
from analysis.results import (
    CLASS_PARTITIONED_LINEAR_EVALUATOR_TO_CKPT_DIR,
    CLASS_PARTITIONED_FINETUNED_EVALUATOR_TO_CKPT_DIR,
)


class Finetuning(BaseModel):
    def __init__(
        self,
        backbone: pl.LightningModule,
        learning_rate: float = 1e-1,
        optimizer: str = "adam",
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
        top_k: Tuple[int, ...] = (1,),
        datamodule=Optional[pl.LightningDataModule],
        track_per_class_accuracy: bool = False,
    ):
        """Pytorch Lightning Module defining behavior for model finetuning.

        Args:
            backbone (pl.LightningModule): model backbone to use
            learning_rate (float, optional): learning rate to use in training. Defaults to 1e-1.
            optimizer (str, optional): optimizer to use. Defaults to "adam".
            momentum (float, optional): momentum to use. Defaults to 0.9.
            weight_decay (float, optional): decay added to weights. Defaults to 1e-4.
            top_k (Tuple[int, ...], optional): which integers (k) to log as 'top_k' accuracy. Defaults to (1,).
            datamodule (_type_, optional): datamodule to use. Defaults to Optional[pl.LightningDataModule].
            track_per_class_accuracy (bool, optional): control for tracking per-class accuracy in addition to aggregate. Defaults to False.
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.track_per_class_accuracy = track_per_class_accuracy

        self.datamodule = datamodule
        self.top_k = top_k
        self.num_classes = None

        # infers num_classes from dataloader
        self._setup_loader_names()
        if self.top_k:
            self.setup_accuracy_metrics()
            if track_per_class_accuracy:
                self.setup_per_class_accuracy()

        self.backbone = backbone
        self.check_backbone()
        self.linear_classifier = torch.nn.Linear(
            self.backbone.feature_dim, self.num_classes
        )

    def check_backbone(self):
        assert hasattr(
            self.backbone, "feature_dim"
        ), "backbone missing feature_dim attribute"
        assert hasattr(self.backbone, "forward"), "backbone missing forward method"

    def forward(self, x):
        features = self.backbone(x)
        logits = self.linear_classifier(features)
        return logits

    def unpack_batch(self, batch: Tensor) -> Tuple[Tensor, Tensor]:
        x, y, _ = batch
        return x, y

    def shared_step(self, batch: Tensor, stage: str = "train"):
        x, y = self.unpack_batch(batch)
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
            if self.track_per_class_accuracy:
                self.log_per_class_accuracy(y_hat, y, stage, k)
        return loss

    def log_per_class_accuracy(self, y_hat, y, stage: str, k: int):
        acc_name = f"{stage}_top_{k}_per_class_accuracy"
        acc_func = getattr(self, acc_name)
        acc_func(y_hat.softmax(dim=-1), y)
        self.log_dict(acc_func.compute(), add_dataloader_idx=False)

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
        raise ValueError(f"optimizer {self.optimizer} not implemented")


class LinearEval(Finetuning):
    """Pytorch lightning module defining behavior for linearEval mode."""

    def on_train_epoch_start(self) -> None:
        self.backbone.eval()

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
        logits = self.linear_classifier(features)
        return logits


class StorePredictions(BaseModel):
    def __init__(
        self,
        backbone: pl.LightningModule,
        datamodule: pl.LightningDataModule,
        ckpt_dir: Optional[str] = None,
        eval_type: str = "linear_eval",
        save_dir: str = "/checkpoint/marksibrahim/results/robustness-limits/predictions",
    ):
        """Pytorch Lightning module defining behavior for storing predictions (stores prediction, true class, and loader type)

        Args:
            backbone (pl.LightningModule): model backbone to use
            datamodule (pl.LightningDataModule): datamodule to evaluate on
            ckpt_dir (Optional[str], optional): path to checkpoint directory for model weights. Defaults to None.
            eval_type (str, optional): evaluation mode to use. Defaults to "linear_eval".
            save_dir (str, optional): path to directory for saving. Defaults to "/checkpoint/marksibrahim/results/robustness-limits/predictions".
        """
        super().__init__()

        self.backbone = backbone
        self.ckpt_dir = ckpt_dir
        self.eval_type = eval_type
        self.datamodule = datamodule
        self.save_dir = save_dir

        if ckpt_dir is None:
            self.ckpt_dir = self.find_checkpoint_for_class_partitioned_evaluator()
        self.evaluator = self.load_evaluator()

        # infer from datamodule
        self.num_classes = None
        # infers num_classes from dataloader
        self._setup_loader_names()

        # predictions stored in results dict (from parent class)
        # "keys: "pred", "true_class", "loader_name"
        self.results["predictions"] = []

    def find_checkpoint_for_class_partitioned_evaluator(self):
        model_class_name = self.backbone.__class__.__name__
        if self.eval_type == "linear_eval":
            ckpt_dir = CLASS_PARTITIONED_LINEAR_EVALUATOR_TO_CKPT_DIR[model_class_name]
        elif self.eval_type == "finetuning":
            ckpt_dir = CLASS_PARTITIONED_FINETUNED_EVALUATOR_TO_CKPT_DIR[
                model_class_name
            ]
        else:
            raise ValueError(f"{self.eval_type} not supported")
        print("loaded checkpoint from ", ckpt_dir)
        return ckpt_dir

    def load_evaluator(self):
        if self.eval_type == "linear_eval":
            ckpt = find_existing_checkpoint(self.ckpt_dir)
            evaluator = LinearEval.load_from_checkpoint(
                ckpt, backbone=self.backbone, datamodule=self.datamodule
            )
        elif self.eval_type == "finetuning":
            ckpt = find_existing_checkpoint(self.ckpt_dir)
            evaluator = Finetuning.load_from_checkpoint(
                ckpt, backbone=self.backbone, datamodule=self.datamodule
            )
        else:
            raise ValueError(f"{self.eval_type} not supported")
        return evaluator

    def predict(self, batch: Tensor, stage: str = "train") -> Tuple[Tensor, Tensor]:
        x, y, _ = batch
        # logits
        y_hat = self.evaluator(x)
        preds = torch.argmax(y_hat, dim=1)
        return preds, y

    def save_preds(self, preds: Tensor, true_classes: Tensor, loader_name: str):
        preds = preds.tolist()
        true_classes = true_classes.tolist()
        prediction_records = [
            {"pred": p, "true_class": t, "loader_name": loader_name}
            for p, t in zip(preds, true_classes)
        ]
        current_records = self.results["predictions"]
        new_records = current_records + prediction_records
        self.results["predictions"] = new_records

    def validation_step(self, batch, batch_idx, loader_idx=0):
        loader_name = self.val_loader_names[loader_idx]
        preds, true_classes = self.predict(batch, stage=loader_name)
        self.save_preds(preds, true_classes, loader_name)
        return None

    def on_validation_end(self) -> None:
        self.predictions_to_csv()

    def predictions_to_csv(self):
        predictions = self.results["predictions"]
        df = pd.DataFrame.from_records(predictions)
        save_path = os.path.join(
            self.save_dir,
            f"{self.eval_type}_{self.backbone.__class__.__name__}_predictions.csv",
        )
        df.to_csv(save_path, index=False)
