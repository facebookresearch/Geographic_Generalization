from models.classifier_model import ClassifierModule


class MLPMixerClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mixer_b16_224",
        learning_rate: float = 1e-4,
        optimizer: str = "adam",
        checkpoint_url: str = "https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_mixer_b16_224-76587d61.pth",
    ):
        super().__init__(timm_name=timm_name, checkpoint_url=checkpoint_url)
