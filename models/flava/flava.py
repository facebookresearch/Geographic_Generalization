"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from datasets.imagenet_classes import IMAGENET_CLASSES
from datasets.geode import GEODE_CLASSES_TO_IMAGENET_CLASSES
import torch
from models.classifier_model import ClassifierModule
from transformers import BertTokenizer, FlavaModel, FlavaImageProcessor
from PIL import Image
from datasets.dollarstreet import MAPPING


class FLAVAClassifierModule(ClassifierModule):
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
            print("Using Imagenet labels for FLAVA")
        elif dataset_to_use_for_classes == "Geode":
            self.class_list = list(sorted(GEODE_CLASSES_TO_IMAGENET_CLASSES.keys()))
            print("Using Geode labels for FLAVA")
        elif dataset_to_use_for_classes == "DollarStreet":
            self.class_list = list(sorted(MAPPING.keys()))
            print("Using DollarStreet labels for FLAVA")

        else:
            raise Exception(
                "FLAVAClassifierModule accepts two options for the 'dataset_to_use_for_classes' parameter: Imagenet or Geode"
            )

        super().__init__(
            timm_name=timm_name,
            feature_extraction_layer_index=feature_extraction_layer_index,
            checkpoint_url=checkpoint_url,
            linear_eval=linear_eval,
        )

    def load_model(self):
        model = FlavaModel.from_pretrained("facebook/flava-full").to(self.device)
        self.feature_extractor = FlavaImageProcessor.from_pretrained(
            "facebook/flava-full"
        )
        self.processor = BertTokenizer.from_pretrained("facebook/flava-full")

        self.processed_class_names = self.processor(
            text=[f"a photo of a {x}" for x in self.class_list],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
        ).to(self.device)
        # self.processed_class_names = ["a"]
        print("FLava model loaded...")

        return model

    def forward(self, x):
        # assert x.device == self.model.device

        # # Get text embedding
        text_embeddings = self.model.get_text_features(
            **self.processed_class_names.to(self.model.device)
        ).to(
            self.model.device
        )  # [1000, 77, 768])

        text_embedding = text_embeddings[:, 0, :]  # [1000, 768])
        text_embedding = torch.nn.functional.normalize(text_embedding, dim=-1)

        # # Get image embedding
        image_embeddings = self.model.get_image_features(x)  # [5, 197, 768]
        image_embedding = image_embeddings[:, 0, :]
        image_embedding = torch.nn.functional.normalize(
            image_embedding, dim=-1
        )  # [5, 768]

        # # Generate logits
        # print("OK now multiplying")
        # print(text_embedding.transpose(1,0).shape) #[768, 1000]
        logits = torch.matmul(
            image_embedding, text_embedding.transpose(1, 0)
        )  # [5, 1000]

        # logits = torch.rand(x.shape[0], 1000).to(self.device)
        return logits

    def forward_features(self, x):
        with torch.no_grad():
            self.model.eval()
        self.feature_extraction_layer = ""
        return []

    def load_backbone(self):
        example = torch.rand((1, 3, 224, 224))
        embedding_dim = 0
        self.forward_features(example)
        return None, self.feature_extraction_layer, [], embedding_dim
