from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torch


class ImageDataset(Dataset):
    """Randomly generated image dataset designed to be used with ImageNet models."""

    def __init__(
        self,
        num_samples: int = 5000,
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
        """Randomly generated ('dummy') image dataset designed to be used with ImageNet models.

        Args:
            batch_size (int, optional): _description_. Defaults to 8.
            num_classes (int, optional): _description_. Defaults to 55.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ds = ImageDataset(num_classes=self.num_classes)

        self.train_loader_names = ["train"]
        self.val_loader_names = ["val"]
        self.test_loader_names = ["test_1"]

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)

    def test_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)


if __name__ == "__main__":
    dm = DummyDataModule()
    dm.train_dataloader()
