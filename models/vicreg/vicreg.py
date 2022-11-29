from models.vicreg.architecture import build_convnexttransformer_backbone
import torch
import copy
import pytorch_lightning as pl


class VICRegPretrained21k(pl.LightningModule):
    """Loads VICReg with the ConvNext architecture"""

    WEIGHTS_PATH = "/checkpoint/abardes/vicreg/vicreg_cnxt-H-drop0.1_im22k_ms-2-6_lr5e-05_bs1024_90ep/model.pth"
    PRETRAINING_DATASET = "imagenet22k"

    def __init__(self):
        super().__init__()
        self.feature_dim = 2048
        self.model = self.load_backbone()

    @staticmethod
    def load_from_state_dict(model, state_dict, prefix, new_suffix):
        state_dict = copy.deepcopy(state_dict)
        state_dict = {
            k.replace(prefix, new_suffix): v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        for k, v in model.state_dict().items():
            if k not in list(state_dict):
                print('key "{}" could not be found in provided state dict'.format(k))
            elif state_dict[k].shape != v.shape:
                print(
                    'key "{}" is of different shape in model and provided state dict {} vs {}'.format(
                        k, v.shape, state_dict[k].shape
                    )
                )
                state_dict[k] = v
        msg = model.load_state_dict(state_dict, strict=False)
        print("Load pretrained model with msg: {}".format(msg))
        return model

    def load_backbone(self):
        model, embedding_dim = build_convnexttransformer_backbone(
            size="H", dataset=self.PRETRAINING_DATASET, drop_path=0.1, dim=-1
        )
        ckpt = torch.load(
            self.WEIGHTS_PATH,
            map_location="cpu",
        )
        model = self.load_from_state_dict(
            model, ckpt["model"], prefix="module.online.net.", new_suffix=""
        )
        return model

    def forward(self, x):
        return self.model(x)


class VICRegPretrained1k(VICRegPretrained21k):
    """Loads VICReg with the ConvNext architecture"""

    WEIGHTS_PATH = "/checkpoint/abardes/vicreg/vicreg_cnxt-H_opt-adamw_lr0.0001_bs512_1000ep/model.pth"
    PRETRAINING_DATASET = "imagenet1k"