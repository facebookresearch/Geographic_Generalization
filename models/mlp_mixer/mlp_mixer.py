from models.classifier_model import ClassifierModule


class MLPMixerClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mixer_b16_224",
        checkpoint_url: str = "",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
