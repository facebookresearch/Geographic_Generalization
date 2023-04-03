from models.classifier_model import ClassifierModule


class RegNet2ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_002",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet4ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_004",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet6ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_006",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet8ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_008",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet16ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_016",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet32ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_032",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet40ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_040",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet64ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_064",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet120ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_120",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)


class RegNet320ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnety_320",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
