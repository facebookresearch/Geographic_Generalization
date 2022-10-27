from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import supporters


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
        self.fov = {
            "pose_x": 0,
            "pose_y": 0,
            "pose_z": 0,
            "image_path": "great_path.png",
        }

    def __getitem__(self, index):
        return self.x[index], self.labels[index]

    def __len__(self):
        return self.len


class DummyDataSingleFactorDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_classes: int = 55, num_workers = 80):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ds = ImageDataset(num_classes=self.num_classes)

        self.train_loader_names = ["train"]
        self.val_loader_names = ["val"]

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

if __name__ == "__main__":
    dm = DummyDataSingleFactorDataModule()
    dm.train_dataloader()
