import pytorch_lightning as pl
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization
from torchvision import transforms as transform_lib
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from typing import Callable
import os
from torchvision.transforms import InterpolationMode

# resnet50: bilinear, 235
# resnet101: bicubic, 235
# mlpmixer: bicubic, 256 resize, normalize with all means and std as 0.5
# vit: bicubic, 248, normalization at 0.5s
# vitlarge: bicubic


class ImageDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/datasets01/imagenet_full_size/061417",
        batch_size: int = 32,
        num_workers: int = 8,
        image_size: int = 224,
        test_dir_name: str = "test",
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
        augmentations = self.val_transform()
        data_loader = self._create_dataloader("test", augmentations)
        return data_loader

    def _get_dataset(self, path, augmentations):
        return ImageFolder(path, augmentations)

    def _create_dataloader(self, stage: str, augmentations: transform_lib.Compose):
        path = os.path.join(self.data_dir, stage)
        shuffle = True if stage == "train" else False
        dataset = self._get_dataset(path, augmentations)
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
                transform_lib.Resize(
                    int(248),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transform_lib.CenterCrop(self.image_size),
                transform_lib.ToTensor(),
                transform_lib.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # imagenet_normalization(),
            ]
        )
        print("using val transform\n", preprocessing)
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
