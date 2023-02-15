from models.classifier_model import ClassifierModule


class BeitClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "beit_base_patch16_224",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
