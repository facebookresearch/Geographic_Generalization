"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from models.classifier_model import ClassifierModule


class MobileVitSClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mobilevit_s",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileVitXSClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mobilevit_xs",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768


class MobileVitXXSClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "mobilevit_xxs",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )

    def load_backbone(self):
        # TODO: getting an erro with this
        return None, None, None, 768
