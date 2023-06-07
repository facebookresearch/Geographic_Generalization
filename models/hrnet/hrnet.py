from models.classifier_model import ClassifierModule


class HRNet64ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "hrnet_w64",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class HRNet48ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "hrnet_w48",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class HRNet40ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "hrnet_w40",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class HRNet44ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "hrnet_w44",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class HRNet32ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "hrnet_w32",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class HRNet30ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "hrnet_w30",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class HRNet18ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "hrnet_w18",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class HRNet18SmallClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "hrnet_w18_small",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
