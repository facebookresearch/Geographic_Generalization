from models.classifier_model import ClassifierModule
from timm import create_model


class ConvNextlassifierModule(ClassifierModule):
    """ConvNext based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py"""

    def __init__(
        self,
        timm_name: str = "convnext_base_in22k",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)

    def load_backbone(self):
        """loads ConvNext model"""
        model = create_model(self.timm_name, pretrained=True, num_classes=1000)
        return model
