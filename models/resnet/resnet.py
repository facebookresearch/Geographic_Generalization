import torch.nn.functional as F
from models.classifier_model import ClassifierModule
import torchvision
import antialiased_cnns
import torch


class ResNet18ClassifierModule(ClassifierModule):
    """Weights from Resnet Strikes Back.
    Paper: https://arxiv.org/abs/2110.00476
    Weights: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-rsb-weights
    """

    def __init__(
        self,
        timm_name: str = "resnet18",
        checkpoint_url: str = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet18_a1_0-d63eafa0.pth",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class ResNet34ClassifierModule(ClassifierModule):
    """Weights from Resnet Strikes Back.
    Paper: https://arxiv.org/abs/2110.00476
    Weights: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-rsb-weights
    """

    def __init__(
        self,
        timm_name: str = "resnet34",
        checkpoint_url: str = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet34_a1_0-46f8f793.pth",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class ResNet50ClassifierModule(ClassifierModule):
    """Resnet 50 with recipe from https://arxiv.org/abs/2110.00476"""

    def __init__(
        self,
        timm_name: str = "resnet50",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50_a1_0-14fe96d1.pth",
        linear_eval: bool = False,
    ):
        """Loads ResNet-50 weights from ResNet Strikes back paper.
        Weights based on the A1 training recipe aiming for the best performance
        with 600 epochs (top 1 accuracy of 80.4).
        https://arxiv.org/abs/2110.00476
        Weights from https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-rsb-weights
        """
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class ResNet101ClassifierModule(ClassifierModule):
    """Weights from Resnet Strikes Back.
    Paper: https://arxiv.org/abs/2110.00476
    Weights: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-rsb-weights
    """

    def __init__(
        self,
        timm_name: str = "resnet101",
        checkpoint_url: str = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1_0-cdcb52a9.pth",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class ResNet152ClassifierModule(ClassifierModule):
    """Weights from Resnet Strikes Back.
    Paper: https://arxiv.org/abs/2110.00476
    Weights: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.1-rsb-weights
    """

    def __init__(
        self,
        timm_name: str = "resnet152",
        checkpoint_url: str = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet152_a1_0-2eee8a7a.pth",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


##### Anti aliased #####


class ResNet18AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.resnet18(pretrained=True)
        return model

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


class ResNet34AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.resnet34(pretrained=True)
        return model

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


class ResNet50AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.resnet50(pretrained=True)
        return model

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


class ResNet101AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.resnet101(pretrained=True)
        return model

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


class ResNet152AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.resnet152(pretrained=True)
        return model

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
