"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
# eva_giant_patch14_224.clip_ft_in1k
from models.classifier_model import ClassifierModule


class EvaCLIPClassifierModule(ClassifierModule):
    """Weights from CVPR 2023 EVA-CLIP Paper, SOTA on Imagenet.
    Paper: https://arxiv.org/abs/2211.07636
    Weights: https://github.com/huggingface/pytorch-image-models/releases/tag/v0.8.2dev0
    """

    def __init__(
        self,
        timm_name: str = "eva_giant_patch14_224.clip_ft_in1k",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
