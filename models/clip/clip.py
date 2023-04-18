from typing import Tuple, Optional, List
from transformers import CLIPModel, CLIPProcessor
import pytorch_lightning as pl
import torch
from torch import Tensor
import torch.nn.functional as F

import open_clip
from open_clip import tokenizer
from datasets.imagenet_classes import IMAGENET_CLASSES
from datasets.geode import GEODE_CLASSES_TO_IMAGENET_CLASSES
from models.classifier_model import ClassifierModule


class OpenCLIPBaseClassifierModule(ClassifierModule):
    """"""

    def __init__(
        self,
        timm_name: str = "",
        feature_extraction_layer_index=-2,
        checkpoint_url: str = "",
        linear_eval: bool = False,
        dataset_to_use_for_classes: str = "Imagenet",
    ):
        if dataset_to_use_for_classes == "Imagenet":
            self.class_list = IMAGENET_CLASSES
            print("Using Imagenet labels for CLIP")
        elif dataset_to_use_for_classes == "Geode":
            self.class_list = list(sorted(GEODE_CLASSES_TO_IMAGENET_CLASSES.keys()))
            print("Using Geode labels for CLIP")

        # based on https://colab.research.google.com/github/mlfoundations/open_clip/blob/master/docs/Interacting_with_open_clip.ipynb#scrollTo=C4S__zCGy2MT
        self.CLASS_NAME_PROMPTS = [f"This is a photo of a {c}" for c in self.class_list]
        super().__init__(
            timm_name=timm_name,
            feature_extraction_layer_index=feature_extraction_layer_index,
            checkpoint_url=checkpoint_url,
            linear_eval=linear_eval,
        )

    def load_model(self):
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

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
            self.model.eval()
            self.feature_extraction_layer = "image_embeds"
            text_input = self.processor(
                text=self.CLASS_NAME_PROMPTS, return_tensors="pt", padding=True
            ).to(self.device)

            image_input = {"pixel_values": x}
            all_inputs = {**text_input, **image_input}
            outputs = self.model(**all_inputs)
        return outputs[self.feature_extraction_layer]

    def load_backbone(self):
        example = torch.rand((1, 3, 224, 224))
        output = self.forward_features(example)
        embedding_dim = output.shape[1]
        return None, self.feature_extraction_layer, [], embedding_dim


class OpenCLIPBaseWithoutEmbeddingClassifierModule(OpenCLIPBaseClassifierModule):
    """Vit-B32, pretrained on LAION-400M. Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32-quickgelu", pretrained="laion400m_e32"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32-quickgelu")
        print("using preprocess", self.preprocess)
        return model

    def forward(self, x):
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.model.eval()
            self.text_tokens = self.tokenizer(self.CLASS_NAME_PROMPTS).to(self.device)
            self.text_features = self.model.encode_text(self.text_tokens).float()
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            image_features = self.model.encode_image(x).float()

        logits = 100.0 * image_features @ self.text_features.T
        return logits

    def forward_features(self, x):
        with torch.no_grad(), torch.cuda.amp.autocast():
            self.model.eval()
            self.text_tokens = self.tokenizer(self.CLASS_NAME_PROMPTS).to(self.device)
            self.text_features = self.model.encode_text(self.text_tokens).float()
            self.text_features /= self.text_features.norm(dim=-1, keepdim=True)
            image_features = self.model.encode_image(x).float()
            return image_features

    def load_backbone(self):
        example = torch.rand((1, 3, 224, 224))
        output = self.forward_features(example)
        embedding_dim = output.shape[1]
        return None, "", [], embedding_dim


class CLIPCoCaVit14ClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "coca_ViT-L-14"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        return model


class CLIPCoCaVit32ClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "coca_ViT-B-32"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model


class CLIPConvNextLaion2BAClassifierModule(
    OpenCLIPBaseWithoutEmbeddingClassifierModule
):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "convnext_base_w", "laion_aesthetic_s13b_b82k"
        )
        self.tokenizer = open_clip.get_tokenizer("convnext_base_w")
        return model


class CLIPConvNextLaion2BAugClassifierModule(
    OpenCLIPBaseWithoutEmbeddingClassifierModule
):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "convnext_base_w", "laion2b_s13b_b82k_augreg"
        )
        self.tokenizer = open_clip.get_tokenizer("convnext_base_w")
        return model


class CLIPConvNextLaion2BClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "convnext_base_w", "laion2b_s13b_b82k"
        )
        self.tokenizer = open_clip.get_tokenizer("convnext_base_w")
        return model


class CLIPConvNextLargeLaion2BClassifierModule(
    OpenCLIPBaseWithoutEmbeddingClassifierModule
):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "convnext_large_d", "laion2b_s26b_b102k_augreg"
        )
        self.tokenizer = open_clip.get_tokenizer("convnext_large_d")
        return model


class CLIPConvNextXLargeLaion2BAugClassifierModule(
    OpenCLIPBaseWithoutEmbeddingClassifierModule
):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "convnext_xxlarge", "laion2b_s34b_b82k_augreg"
        )
        self.tokenizer = open_clip.get_tokenizer("convnext_xxlarge")
        return model


class CLIPConvNextXLargeLaion2BRewindClassifierModule(
    OpenCLIPBaseWithoutEmbeddingClassifierModule
):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "convnext_xxlarge", "laion2b_s34b_b82k_augreg_rewind"
        )
        self.tokenizer = open_clip.get_tokenizer("convnext_xxlarge")
        return model


class CLIPConvNextXLargeLaion2BSoupClassifierModule(
    OpenCLIPBaseWithoutEmbeddingClassifierModule
):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "convnext_xxlarge", "laion2b_s34b_b82k_augreg_soup"
        )
        self.tokenizer = open_clip.get_tokenizer("convnext_xxlarge")
        return model


class CLIPResnet50CC12MClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "RN50", "cc12m"
        )
        self.tokenizer = open_clip.get_tokenizer("RN50")
        return model


class CLIPResnet50OpenAIClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "RN50", "openai"
        )
        self.tokenizer = open_clip.get_tokenizer("RN50")
        return model


class CLIPResnet50YFCCClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "RN50", "yfcc15m"
        )
        self.tokenizer = open_clip.get_tokenizer("RN50")
        return model


class CLIPResnet101OpenAIClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "RN101", "openai"
        )
        self.tokenizer = open_clip.get_tokenizer("RN101")
        return model


class CLIPResnet101YFCCClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "RN101", "yfcc15m"
        )
        self.tokenizer = open_clip.get_tokenizer("RN101")
        return model


class CLIPVit14Laion2BClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", "laion2b_s32b_b82k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        return model


class CLIPVit14Laion400MBClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", "laion400m_e31"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        return model


class CLIPVit14OpenAIClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-L-14", "openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-L-14")
        return model


class CLIPVit16Laion2BClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", "laion2b_s34b_b88k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        return model


class CLIPVit16Laion400MClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", "laion400m_e31"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        return model


class CLIPVit16OpenAIClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-16", "openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-16")
        return model


class CLIPVit32Laion2BClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", "laion2b_s34b_b88k"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model


class CLIPVit32Laion400MClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", "laion400m_e31"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model


class CLIPVit32OpenAIClassifierModule(OpenCLIPBaseWithoutEmbeddingClassifierModule):
    """Based on models trained from https://github.com/mlfoundations/open_clip"""

    def load_model(self):
        model, _, self.preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", "openai"
        )
        self.tokenizer = open_clip.get_tokenizer("ViT-B-32")
        return model
