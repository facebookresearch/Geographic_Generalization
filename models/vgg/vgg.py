from models.classifier_model import ClassifierModule


class Vgg11ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "vgg11",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        return None, "", [], 500


class Vgg13ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "vgg13",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        return None, "", [], 500


class Vgg16ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "vgg16",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        return None, "", [], 500


class Vgg19ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "vgg19",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        return None, "", [], 500
