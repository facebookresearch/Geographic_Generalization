from models.classifier_model import ClassifierModule


class ConvNextSmallClassifierModule(ClassifierModule):
    """ConvNext based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py"""

    def __init__(
        self,
        timm_name: str = "convnext_small",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class ConvNextBaseClassifierModule(ClassifierModule):
    """ConvNext based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py"""

    def __init__(
        self,
        timm_name: str = "convnext_base",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class ConvNextLargeClassifierModule(ClassifierModule):
    """ConvNext based on https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/convnext.py"""

    def __init__(
        self,
        timm_name: str = "convnext_large",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
