from models.classifier_model import ClassifierModule


class RexNet100ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "rexnet_100",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RexNet130ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "rexnet_130",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RexNet150ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "rexnet_150",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RexNet200ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "rexnet_200",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
