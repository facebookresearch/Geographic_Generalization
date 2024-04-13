"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from models.classifier_model import ClassifierModule


class DLA169ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla169",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class DLA102ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla102",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class DLA102xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla102x",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class DLA60xcClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla60x_c",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class DLA60xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla60x",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class DLA60ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla60",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class DLA46xcClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla46x_c",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class DLA46cClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla46_c",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class DLA34ClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "dla34",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
