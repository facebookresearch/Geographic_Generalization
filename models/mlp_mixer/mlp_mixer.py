from models.classifier_model import ClassifierModule


class MLPMixerClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mixer_b16_224",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
