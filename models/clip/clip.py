from typing import Tuple, Optional, List
from transformers import CLIPModel, CLIPProcessor
import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn.functional as F

import open_clip
from open_clip import tokenizer
from datasets.imagenet_classes import IMAGENET_CLASSES
from models.classifier_model import ClassifierModule


class CLIPClassifierModule(ClassifierModule):
    """CLIP zero shot classiifer"""

    def __init__(
        self,
        timm_name: str = "",
        feature_extraction_layer_index=-2,
        checkpoint_url: str = "",
    ):
        super().__init__(
            timm_name=timm_name,
            feature_extraction_layer_index=feature_extraction_layer_index,
            checkpoint_url=checkpoint_url,
        )

    # based on https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb#scrollTo=C4S__zCGy2MT
    CLASS_NAME_PROMPTS = [f"This is a photo of a {c}" for c in IMAGENET_CLASSES]
    text_tokens = tokenizer.tokenize(CLASS_NAME_PROMPTS)

    def load_backbone(self):
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-16")
        self.preprocess = preprocess
        return model

    def forward(self, x):
        text_input = self.processor(
            text=self.CLASS_NAME_PROMPTS, return_tensors="pt", padding=True
        ).to(self.device)
        image_input = {"pixel_values": x}
        all_inputs = {**text_input, **image_input}
        outputs = self.model(**all_inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image

    def forward_features(self, x):
        with torch.no_grad():
            self.feature_extraction_layer = "image_embeds"
            self.processor.eval()
            self.model.eval()
            text_input = self.processor(
                text=self.CLASS_NAME_PROMPTS, return_tensors="pt", padding=True
            ).to(self.device)

            image_input = {"pixel_values": x}
            all_inputs = {**text_input, **image_input}
            outputs = self.model(**all_inputs)
        return outputs[self.feature_extraction_layer]

    def load_feature_extractor(self):
        example = torch.rand((1, 3, 224, 224))
        output = self.forward_features(example)
        embedding_dim = output.shape[1]
        return None, self.feature_extraction_layer, [], embedding_dim


class CLIPOPENAI400MClassifierModule(ClassifierModule):
    """CLIP zero shot classiifer"""

    def __init__(
        self,
        timm_name: str = "",
        feature_extraction_layer_index=-2,
        checkpoint_url: str = "",
    ):
        super().__init__(
            timm_name=timm_name,
            feature_extraction_layer_index=feature_extraction_layer_index,
            checkpoint_url=checkpoint_url,
        )

    # based on https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb#scrollTo=C4S__zCGy2MT
    CLASS_NAME_PROMPTS = [f"This is a photo of a {c}" for c in IMAGENET_CLASSES]

    def load_backbone(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        print("Model loaded")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("processor loaded")
        return model

    def forward(self, x):
        text_input = self.processor(
            text=self.CLASS_NAME_PROMPTS, return_tensors="pt", padding=True
        ).to(self.device)
        image_input = {"pixel_values": x}
        all_inputs = {**text_input, **image_input}
        outputs = self.model(**all_inputs)
        logits_per_image = outputs.logits_per_image
        return logits_per_image

    def forward_features(self, x):
        with torch.no_grad():
            self.feature_extraction_layer = "image_embeds"
            self.processor.eval()
            self.model.eval()
            text_input = self.processor(
                text=self.CLASS_NAME_PROMPTS, return_tensors="pt", padding=True
            ).to(self.device)

            image_input = {"pixel_values": x}
            all_inputs = {**text_input, **image_input}
            outputs = self.model(**all_inputs)
        return outputs[self.feature_extraction_layer]

    def load_feature_extractor(self):
        example = torch.rand((1, 3, 224, 224))
        output = self.forward_features(example)
        embedding_dim = output.shape[1]
        return None, self.feature_extraction_layer, [], embedding_dim


class CLIPLAION400MClassifierModule(CLIPOPENAI400MClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_backbone(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="laion400m_e32"
        )
        return model

    def forward(self, x):
        self.text_tokens = tokenizer.tokenize(self.CLASS_NAME_PROMPTS).to(self.device)
        self.text_features = self.model.encode_text(self.text_tokens).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        image_features = self.model.encode_image(x).float()

        logits = 100.0 * image_features @ self.text_features.T
        return logits

    def forward_features(self, x):
        self.text_tokens = tokenizer.tokenize(self.CLASS_NAME_PROMPTS).to(self.device)
        self.text_features = self.model.encode_text(self.text_tokens).float()
        self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
        image_features = self.model.encode_image(x).float()
        return image_features

    def load_feature_extractor(self):
        example = torch.rand((1, 3, 224, 224))
        output = self.forward_features(example)
        embedding_dim = output.shape[1]
        return None, "", [], embedding_dim


class CLIPLAION2BClassifierModule(CLIPLAION400MClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_backbone(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32",
            pretrained="laion2b_e16",
        )
        return model
