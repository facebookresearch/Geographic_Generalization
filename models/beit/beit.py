from models.classifier_model import ClassifierModule


class BeitBaseClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "beit_base_patch16_224",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class BeitLargeClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "beit_large_patch16_224",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
