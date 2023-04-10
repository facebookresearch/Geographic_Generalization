from models.classifier_model import ClassifierModule


class TinyNetAClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_a",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class TinyNetBClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_b",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class TinyNetCClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_c",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class TinyNetDClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_d",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class TinyNetEClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_e",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
