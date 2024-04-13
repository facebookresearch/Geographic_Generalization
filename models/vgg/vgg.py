"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from models.classifier_model import ClassifierModule
import antialiased_cnns
import torch


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


##### Antialiased #####


class Vgg11AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.vgg11(pretrained=True)
        return model

    def forward_features(self, x):
        with torch.no_grad():
            self.model.eval()
        self.feature_extraction_layer = ""
        return []

    def load_backbone(self):
        example = torch.rand((1, 3, 224, 224))
        embedding_dim = 0
        self.forward_features(example)
        return None, self.feature_extraction_layer, [], embedding_dim


class Vgg13AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.vgg13(pretrained=True)
        return model

    def forward_features(self, x):
        with torch.no_grad():
            self.model.eval()
        self.feature_extraction_layer = ""
        return []

    def load_backbone(self):
        example = torch.rand((1, 3, 224, 224))
        embedding_dim = 0
        self.forward_features(example)
        return None, self.feature_extraction_layer, [], embedding_dim


class Vgg16AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.vgg16(pretrained=True)
        return model

    def forward_features(self, x):
        with torch.no_grad():
            self.model.eval()
        self.feature_extraction_layer = ""
        return []

    def load_backbone(self):
        example = torch.rand((1, 3, 224, 224))
        embedding_dim = 0
        self.forward_features(example)
        return None, self.feature_extraction_layer, [], embedding_dim


class Vgg19AntiAliasedClassifierModule(ClassifierModule):
    """https://github.com/adobe/antialiased-cnns/blob/master/example_usage2.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = antialiased_cnns.vgg19(pretrained=True)
        return model

    def forward_features(self, x):
        with torch.no_grad():
            self.model.eval()
        self.feature_extraction_layer = ""
        return []

    def load_backbone(self):
        example = torch.rand((1, 3, 224, 224))
        embedding_dim = 0
        self.forward_features(example)
        return None, self.feature_extraction_layer, [], embedding_dim


##### Texture Debiased #####

import timm


class Vgg16TextureClassifierModule(ClassifierModule):
    """https://github.com/rgeirhos/texture-vs-shape/blob/master/models/load_pretrained_models.py"""

    def __init__(
        self,
        timm_name: str = "",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_model(self):
        model = timm.create_model("vgg16", False)
        weights = torch.load(
            "/checkpoint/meganrichards/model_weights/vgg16_texturedebiased.tar"
        )["state_dict"]
        print(weights.keys())
        new_weights = {}
        for k in weights.keys():
            if "module" in k:
                new_weights["".join(k.split("module."))] = weights[k]
            else:
                new_weights[k] = weights[k]

        model.load_state_dict(new_weights)
        return model

    def forward_features(self, x):
        with torch.no_grad():
            self.model.eval()
        self.feature_extraction_layer = ""
        return []

    def load_backbone(self):
        example = torch.rand((1, 3, 224, 224))
        embedding_dim = 0
        self.forward_features(example)
        return None, self.feature_extraction_layer, [], embedding_dim
