"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pytorch_lightning as pl
from torchvision import transforms as transform_lib
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Callable
import os
import torchvision.transforms as transforms

IMAGENET_NORMALIZATION = transforms.Normalize(
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
)


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/imagenet/",
        batch_size: int = 32,
        num_workers: int = 8,
        image_size: int = 224,
    ):
        """Pytorch lightning based datamodule with imagenet defaults.

        Args:
            data_dir (str, optional): Path to dataset directory. Defaults to "/datasets01/imagenet_full_size/061417".
            batch_size (int, optional): Batch size to use in datamodule. Defaults to 32.
            num_workers (int, optional): Number of workers to use in the dataloaders. Defaults to 8.
            image_size (int, optional): Side length for image crop. Defaults to 224.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers
        self.__mask = self.make_mask()

    def make_mask(self):
        return None

    @property
    def mask(self):
        return self.__mask

    def train_dataloader(self) -> DataLoader:
        augmentations = self.train_transform()
        data_loader = self._create_dataloader("train", augmentations)
        return data_loader

    def val_dataloader(self) -> DataLoader:
        augmentations = self.val_transform()
        data_loader = self._create_dataloader("val", augmentations)
        return data_loader

    def test_dataloader(self) -> DataLoader:
        augmentations = self.test_transform()
        data_loader = self._create_dataloader("test", augmentations)
        return data_loader

    def _get_dataset(self, path, stage, augmentations):
        return ImageFolder(path, augmentations)

    def _create_dataloader(self, stage: str, augmentations: transform_lib.Compose):
        path = os.path.join(self.data_dir, stage)
        shuffle = True if stage == "train" else False
        dataset = self._get_dataset(path, stage, augmentations)
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
        return data_loader

    def train_transform(self) -> Callable:
        """
        The standard imagenet transforms
        """
        preprocessing = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                IMAGENET_NORMALIZATION,
            ]
        )
        return preprocessing

    def val_transform(self) -> Callable:
        """
        The standard imagenet transforms for validation
        """
        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                IMAGENET_NORMALIZATION,
            ]
        )
        return preprocessing

    def test_transform(self) -> Callable:
        """
        The standard imagenet transforms for validation
        """
        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                IMAGENET_NORMALIZATION,
            ]
        )
        return preprocessing
