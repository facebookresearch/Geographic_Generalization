from models.classifier_model import ClassifierModule


class BeitBaseClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "beit_base_patch16_224",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class BeitLargeClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "beit_large_patch16_224",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
