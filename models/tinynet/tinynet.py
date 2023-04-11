from models.classifier_model import ClassifierModule


class TinyNetAClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_a",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class TinyNetBClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_b",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class TinyNetCClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_c",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class TinyNetDClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_d",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class TinyNetEClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tinynet_e",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
