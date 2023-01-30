import torch.nn.functional as F
from models.classifier_model import ClassifierModule


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

    def forward_features(self, x):
        # output without pooling
        out = self.model.forward_features(x)
        # pooling
        # based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/resnet.py
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out
