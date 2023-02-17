import torch.nn.functional as F
from models.classifier_model import ClassifierModule
import torchvision


class ResNet50dClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "resnet50d",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class ResNet101dClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "resnet101d",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet101d_ra2-2803ffab.pth",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class ResNet18dClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "resnet18d",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/resnet18d_ra2-48a79e06.pth",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class ResNet152dClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)

    def load_backbone(self):
        model = torchvision.models.resnet152(pretrained=True)
        return model
