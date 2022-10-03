from models.base_model import BaseModel, ShapeNetBaseModel
from typing import Tuple, Optional, List
from transformers import CLIPVisionModel, CLIPModel, CLIPProcessor
import pytorch_lightning as pl
import torch
from torch import Tensor
from analysis.analyze_runs import ClassSimilarity
import torch.nn.functional as F
import open_clip
from open_clip import tokenizer


class CLIPPretrained(pl.LightningModule):
    """Loads a pretrained CLIP Model from Hugggin Face"""

    def __init__(
        self,
    ):
        super().__init__()

        self.feature_dim = 768
        self.model = self.load_backbone()

    def load_backbone(self):
        model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
        return model

    def forward(self, x):
        # model expects a dictionary with pixel_values -> tensor
        expected_input = {"pixel_values": x}
        # based on HuggingFace API for extracting features
        feats = self.model(**expected_input).pooler_output
        return feats


class CLIPZeroShotPretrained(BaseModel):
    def __init__(
        self,
        dataset: str = "warehouse",
        datamodule=Optional[pl.LightningDataModule],
        prompt: str = "photo",
        top_k: Tuple[int, ...] = (1,),
    ):
        super().__init__()
        self.dataset = dataset
        self.model, self.processor = self.load_backbone()
        self.prompt = prompt
        self.class_names = self.load_class_names()
        self.class_name_prompts = self.make_prompts()

        self.datamodule = datamodule
        self.top_k = top_k
        self.track_per_class_accuracy = False
        # infer from datamodule
        self.num_classes = None

        # infers num_classes from dataloader
        self._setup_loader_names()
        if self.top_k:
            self.setup_accuracy_metrics()

    def make_prompts(self) -> List[str]:
        if self.prompt == "photo":
            return [f"a photo of a {c}" for c in self.class_names]
        elif self.prompt == "object":
            return [f"{c}, an inanimate object" for c in self.class_names]
        elif self.prompt == "object photo":
            return [f"a photo of a {c}, an inanimate object" for c in self.class_names]
        elif self.prompt == "item or vehicle":
            return [f"{c}, a household item or vehicle" for c in self.class_names]
        elif self.prompt == "green mountains":
            return [
                f"a photo of a {c} floating over green mountains"
                for c in self.class_names
            ]
        elif self.prompt == "sky":
            return [
                f"a photo of a {c} floating in the sky"
                for c in self.class_names
            ]
        raise ValueError(f"{self.prompt} doesn't exist")

    def load_class_names(self) -> List[str]:
        if self.dataset != "warehouse":
            raise ValueError("only warehosue dataset is supported")
        class_similarity = ClassSimilarity()
        class_idx_to_synset = class_similarity.class_idx_to_synset
        synsets = [class_idx_to_synset[i] for i in range(len(class_idx_to_synset))]
        class_names = [class_similarity.synset_to_name[s] for s in synsets]
        return class_names

    def load_backbone(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        return model, processor

    def forward(self, x):
        with torch.no_grad():
            text_input = self.processor(
                text=self.class_name_prompts, return_tensors="pt", padding=True
            ).to(self.device)
            image_input = {"pixel_values": x}
            all_inputs = {**text_input, **image_input}
            outputs = self.model(**all_inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        return probs

    def shared_step(self, batch: Tensor, stage: str = "train"):
        x, y, fov = batch
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

    def validation_step(self, batch, batch_idx, loader_idx=0):
        loader_name = self.val_loader_names[loader_idx]
        loss = self.shared_step(batch, stage=loader_name)
        return loss

    def test_step(self, batch, batch_idx, loader_idx=0):
        loader_name = self.test_loader_names[loader_idx]
        loss = self.shared_step(batch, stage=loader_name)
        return loss


class CLIPLAIONZeroShotPretrained(CLIPZeroShotPretrained):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_backbone(self):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="laion400m_e32"
        )
        return model, preprocess

    def forward(self, x):
        self.text_tokens = tokenizer.tokenize(self.class_name_prompts).to(self.device)
        self.text_features = self.model.encode_text(self.text_tokens).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        image_features = self.model.encode_image(x).float()

        probs = (100.0 * image_features @ self.text_features.T).softmax(dim=-1)
        return probs


class CLIPLAION2BZeroShotPretrained(CLIPLAIONZeroShotPretrained):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_backbone(self):
        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_e16",
        )
        return model, preprocess
