"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from datasets.image_datamodule import ImageDataModule
import pytorch_lightning as pl
from torchvision import transforms as transform_lib
from torch.utils.data import DataLoader


class ImageDataset(Dataset):
    """Randomly generated image dataset designed to be used with ImageNet models."""

    def __init__(
        self,
        num_samples: int = 500,
        num_classes: int = 1000,
    ):
        """
        Args:
            num_samples (int, optional): Number of samples to create in datasets. Defaults to 5000.
            num_classes (int, optional): Number of classes to represent in the dataset. Defaults to 1000.
        """
        self.len = num_samples
        self.x = torch.rand(num_samples, 3, 224, 224)
        self.labels = torch.randint(num_classes, (num_samples,))

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return self.len


class DummyDataModule(ImageDataModule):
    def __init__(
        self,
        batch_size: int = 8,
        num_classes: int = 55,
        image_size: int = 224,
        num_samples: int = 500,
    ):
        """Randomly generated ('dummy') image dataset designed to be used with ImageNet models.

        Args:
            batch_size (int, optional): _description_. Defaults to 8.
            num_classes (int, optional): _description_. Defaults to 55.
        """
        super().__init__(
            data_dir="",
            batch_size=batch_size,
            image_size=image_size,
        )
        self.num_classes = num_classes
        self.num_samples = num_samples

    def _create_dataloader(self, stage: str, augmentations: transform_lib.Compose):
        if stage == "train":
            shuffle = True
        else:
            shuffle = False
        dataset = ImageDataset(
            num_classes=self.num_classes, num_samples=self.num_samples
        )
        data_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            pin_memory=True,
            num_workers=self.num_workers,
            shuffle=shuffle,
        )
        return data_loader


if __name__ == "__main__":
    dm = DummyDataModule()
    dm.train_dataloader()
