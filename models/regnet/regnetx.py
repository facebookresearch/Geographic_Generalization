"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from models.classifier_model import ClassifierModule


class RegNet2xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_002",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet4xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_004",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet6xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_006",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet8xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_008",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet16xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_016",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet32xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_032",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet40xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_040",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet64xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_064",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet120xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_120",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )


class RegNet320xClassifierModule(ClassifierModule):
    def __init__(
        self,
        timm_name: str = "regnetx_320",
        checkpoint_url: str = "",
        linear_eval: bool = False,
    ):
        super().__init__(
            timm_name=timm_name, checkpoint_url=checkpoint_url, linear_eval=linear_eval
        )
