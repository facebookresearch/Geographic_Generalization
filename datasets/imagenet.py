import os
import pytorch_lightning as pl
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms as transform_lib
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Callable


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/datasets01/imagenet_full_size/061417",
        batch_size: int = 32,
        num_workers=8,
        image_size=224,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.image_size = image_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        augmentations = self.train_transform()
        data_loader = self._create_dataloader("train", augmentations)
        return data_loader

    def val_dataloader(self) -> DataLoader:
        augmentations = self.val_transform()
        data_loader = self._create_dataloader("val", augmentations)
        return data_loader

    def _create_dataloader(self, stage: str, augmentations: transform_lib.Compose):
        path = os.path.join(self.data_dir, stage)
        shuffle = True if stage == "train" else False
        dataset = ImageFolder(path, augmentations)
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
