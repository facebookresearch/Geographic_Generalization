import os
import pytorch_lightning as pl
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms as transform_lib
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Callable
from datasets.image_datamodule import ImageDataModule


class ImageNetDataModule(ImageDataModule):
    def __init__(
        self,
        data_dir: str = "/datasets01/imagenet_full_size/061417",
        batch_size: int = 32,
        num_workers=8,
        image_size=224,
    ):
        """Pytorch lightning based datamodule for ImageNet dataset.

        Args:
            data_dir (str, optional): Path to imagenet dataset directory. Defaults to "/datasets01/imagenet_full_size/061417".
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

    def train_transform(self) -> Callable:
        """
        The standard imagenet transforms
        """
        preprocessing = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                imagenet_normalization(),
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
                imagenet_normalization(),
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
                imagenet_normalization(),
            ]
        )
        return preprocessing
