import torch.nn.functional as F
from models.classifier_model import ClassifierModule


class ResNet50dClassifierModule(ClassifierModule):
    """Resnet 50 with recipe from https://arxiv.org/abs/2110.00476"""

    def __init__(
        self,
        timm_name: str = "resnet50d",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet50d_a1_0-e20cff14.pth",
    ):
        """Loads ResNet-50 weights from ResNet Strikes back paper.
        Weights based on the A1 training recipe aiming for the best performance
        with 600 epochs.
        https://arxiv.org/abs/2110.00476
        Weights from https://github.com/rwightman/pytorch-image-models/releases/tag/v0.1-rsb-weights
        """
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class ResNet101dClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "resnet101",
        checkpoint_url: str = "https://github.com/huggingface/pytorch-image-models/releases/download/v0.1-rsb-weights/resnet101_a1_0-cdcb52a9.pth",
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
