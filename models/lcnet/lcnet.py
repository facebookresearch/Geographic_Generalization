from models.classifier_model import ClassifierModule


class Lcnet50ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "lcnet_050.ra2_in1k",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class Lcnet75ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "lcnet_075.ra2_in1k",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class Lcnet100ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "lcnet_100.ra2_in1k",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768
