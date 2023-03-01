from models.classifier_model import ClassifierModule


class VitClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "vit_base_patch16_224",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class VitLargeClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "vit_large_patch16_224",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
