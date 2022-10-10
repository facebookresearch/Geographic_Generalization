from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch
from pytorch_lightning.trainer import supporters


class ImageDataset(Dataset):
    """Generates a dummy dataset 
    """

    def __init__(
        self,
        num_samples: int = 5000,
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


class DummyDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_classes: int = 55):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ds = ImageDataset(num_classes=self.num_classes)

        self.train_loader_names = ["train"]
        self.val_loader_names = ["val_canonical", "val_diverse_2d", "val_diverse_3d"]
        self.test_loader_names = ["test_1", "test_2"]

        self.train_prop_to_vary = 0.5

    def train_dataloader(self):
        loaders = {
            "train": DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)
        }
        return supporters.CombinedLoader(loaders)

    def val_dataloader(self):
        return [
            DataLoader(self.ds, batch_size=self.batch_size, num_workers=1),
            DataLoader(self.ds, batch_size=self.batch_size, num_workers=1),
            DataLoader(self.ds, batch_size=self.batch_size, num_workers=1),
        ]

    def test_dataloader(self):
        return [
            DataLoader(self.ds, batch_size=self.batch_size, num_workers=1),
            DataLoader(self.ds, batch_size=self.batch_size, num_workers=1),
        ]


if __name__ == "__main__":
    dm = DummyDataModule()
    dm.train_dataloader()
