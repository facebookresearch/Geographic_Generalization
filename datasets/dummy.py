from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import supporters
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms as transform_lib
from typing import Callable
from custom_augmentations import *

class ImageDataset(Dataset):
    """Generates a dummy dataset"""

    def __init__(
        self,
        num_samples: int = 10,
        num_classes: int = 1000,
    ):
        """
        Args:
            size: tuple
            num_samples: number of samples
        """
        self.len = num_samples
        self.x = torch.rand(num_samples, 3, 224, 224)
        self.labels = torch.randint(num_classes, (num_samples,))
        self.image_size=224

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return self.len


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_classes: int = 55, num_workers = 80, image_size = 224):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ds = ImageDataset(num_classes=self.num_classes)
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
        # path = os.path.join(self.data_dir, stage)
        # shuffle = True if stage == "train" else False
        # dataset = ImageFolder(path, augmentations)
        # data_loader = DataLoader(
        #     dataset,
        #     batch_size=self.batch_size,
        #     pin_memory=True,
        #     num_workers=self.num_workers,
        #     shuffle=shuffle,
        # )
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=self.num_workers)
    
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

class DummyAugmentedDataModule(DummyDataModule):
    """
    This module will handle overwriting the transform functions to add custom augmentations.
    Args: 
        - sets_to_augment: list of strings defining the data partitions to apply augmentation to. Entries must match those 
                            used in create_dataloder calls above ("train", "val", "test)
        - augmentation_name: string determining the custom augmentation to apply. Each entry must match case with an object in custom_augmentations.py. E.g. ColorJitter, not colorjitter 
        - augmentaiton_args: dictionary of argument/value pairs to be unpacked into the custom transform. 
    """
    def __init__(self, batch_size: int = 8, num_classes: int = 55, num_workers = 80, image_size = 224, sets_to_augment = "train", augmentation_name = "ColorJitter", **augmentation_args):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ds = ImageDataset(num_classes=self.num_classes)
        self.image_size = image_size
        self.num_workers = num_workers

        self.sets_to_augment = sets_to_augment
        augment_object = globals()[augmentation_name]
        self.custom_transform = augment_object(**augmentation_args)


    def train_transform(self) -> Callable:
        """
        The standard imagenet transforms
        """
        # TODO: add custom transform object, and logic to apply it or not
        preprocessing = transform_lib.Compose(
            [
                transform_lib.RandomResizedCrop(self.image_size),
                transform_lib.RandomHorizontalFlip(),
                transform_lib.ToTensor(),
                imagenet_normalization(),
                self.custom_transform(),
            ]
        )

        return preprocessing

    def val_transform(self) -> Callable:
        """
        The standard imagenet transforms for validation
        """
        # TODO: add custom transform object, and logic to apply it or not
        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                imagenet_normalization(),
                self.custom_transform(),
            ]
        )
        return preprocessing
    
    def test_transform(self) -> Callable:
        """
        The standard imagenet transforms for validation
        """
        # TODO: add custom transform object, and logic to apply it or not
        preprocessing = transform_lib.Compose(
            [
                transform_lib.Resize(self.image_size + 32),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                imagenet_normalization(),
                self.custom_transform(),
            ]
        )
        return preprocessing

    
if __name__ == "__main__":
    dm = DummyDataModule()
    dm.train_dataloader()
