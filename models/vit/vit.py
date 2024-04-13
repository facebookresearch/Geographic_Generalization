"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from models.classifier_model import ClassifierModule


class VitClassifierModule(ClassifierModule):
    """timm/vit_base_patch16_224.augreg2_in21k_ft_in1k, https://github.com/google-research/vision_transformer"""

    def __init__(
        self,
        timm_name: str = "vit_base_patch16_224",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class VitLargeClassifierModule(ClassifierModule):
    """vit_large_patch16_224.augreg_in21k_ft_in1k in timm,  https://github.com/google-research/vision_transformer"""

    def __init__(
        self,
        timm_name: str = "vit_large_patch16_224",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
