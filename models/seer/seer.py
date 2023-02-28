from transformers import RegNetForImageClassification
from models.classifier_model import ClassifierModule
from torch.nn import AdaptiveAvgPool2d
import torch


class Seer320ClassifierModule(ClassifierModule):
    """SEER Models from Hugging Face"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "facebook/regnet-y-320-seer-in1k",
    ):
        super().__init__(
            timm_name=timm_name,
            checkpoint_url=checkpoint_url,
        )

    def load_backbone(self):
        model = RegNetForImageClassification.from_pretrained(self.checkpoint_url)
        return model

    def forward(self, x):
        return self.model(x).logits

    def forward_features(self, x):
        output = self.model.forward(x, output_hidden_states=True, return_dict=True)[
            "hidden_states"
        ][-1]
        pool = AdaptiveAvgPool2d(output_size=(1, 1))
        feat = torch.flatten(pool(output), start_dim=1, end_dim=-1)
        return feat

    def load_feature_extractor(self):
        example = torch.rand((1, 3, 224, 224))
        output = self.forward_features(example)
        embedding_dim = output.shape[1]
        return None, "", [], embedding_dim


class Seer640ClassifierModule(Seer320ClassifierModule):
    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "facebook/regnet-y-640-seer-in1k",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class Seer10bClassifierModule(Seer320ClassifierModule):
    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "facebook/regnet-y-10b-seer-in1k",
    ):
        super().__init__(
            timm_name=timm_name,
            checkpoint_url=checkpoint_url,
        )


class Seer1280ClassifierModule(Seer320ClassifierModule):
    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "facebook/regnet-y-1280-seer-in1k",
    ):
        super().__init__(
            timm_name=timm_name,
            checkpoint_url=checkpoint_url,
        )