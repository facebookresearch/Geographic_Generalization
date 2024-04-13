"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from datasets.image_datamodule import ImageDataModule


class ImageNetSketchDataModule(ImageDataModule):
    def __init__(
        self,
        data_dir: str = "data/imagenet-sketch",
        batch_size: int = 32,
        num_workers=8,
        image_size=224,
    ):
        """Pytorch lightning based datamodule for ImageNet Sketch dataset, found here: https://github.com/HaohanWang/ImageNet-Sketch

        Args:
            data_dir (str, optional): Path to imagenet dataset directory. Defaults to "/checkpoint/meganrichards/datasets/imagenet_sketch/".
            batch_size (int, optional): Batch size to use in datamodule. Defaults to 32.
            num_workers (int, optional): Number of workers to use in the dataloaders. Defaults to 8.
            image_size (int, optional): Side length for image crop. Defaults to 224.
        """

        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )
