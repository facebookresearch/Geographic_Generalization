from models.classifier_model import ClassifierModule
from datasets.imagenet_classes import IMAGENET_CLASSES
from datasets.geode import GEODE_CLASSES_TO_IMAGENET_CLASSES
from lavis.models import load_model_and_preprocess
import torch


class BLIPClassifierModule(ClassifierModule):
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
            print("Using Imagenet labels for BLIP")
        elif dataset_to_use_for_classes == "Geode":
            self.class_list = list(sorted(GEODE_CLASSES_TO_IMAGENET_CLASSES.keys()))
            print("Using Geode labels for BLIP")

        super().__init__(
            timm_name=timm_name,
            feature_extraction_layer_index=feature_extraction_layer_index,
            checkpoint_url=checkpoint_url,
            linear_eval=linear_eval,
        )

    def load_model(self):
        model, vis_processors, txt_processors = load_model_and_preprocess(
            "albef_feature_extractor",
            model_type="base",
            is_eval=True,
            device=self.device,
        )
        self.processor = txt_processors
        self.processed_class_names = [
            self.processor["eval"](cls_nm) for cls_nm in self.class_list
        ]

        return model

    def forward(self, x):
        sample = {"image": x, "text_input": self.processed_class_names}

        image_features = self.model.extract_features(
            sample, mode="image"
        ).image_embeds_proj[:, 0]
        text_features = self.model.extract_features(
            sample, mode="text"
        ).text_embeds_proj[:, 0]

        sims = (image_features @ text_features.t())[0] / self.model.temp
        return sims

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
