from models.classifier_model import ClassifierModule


class MobileNetSmallMin100ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tf_mobilenetv3_small_minimal_100",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileNetSmall75ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tf_mobilenetv3_small_075",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileNetSmall100ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tf_mobilenetv3_small_100",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileNetLargeMin100ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tf_mobilenetv3_large_minimal_100",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileNetLarge75ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tf_mobilenetv3_large_075",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileNetLarge100ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "tf_mobilenetv3_large_100",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


############ Lamb ####################


class MobileNetLamb100ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mobilenetv3_small_100.lamb_in1k",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileNetLamb75ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mobilenetv3_small_075.lamb_in1k",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileNetLamb50ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mobilenetv3_small_050.lamb_in1k",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768
