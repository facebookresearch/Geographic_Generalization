"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from models.classifier_model import ClassifierModule


class RexNet100ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "rexnet_100",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RexNet130ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "rexnet_130",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RexNet150ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "rexnet_150",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RexNet200ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "rexnet_200",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
