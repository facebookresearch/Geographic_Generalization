from torch.utils.data import Dataset, DataLoader
from pl_bolts.datasets import DummyDataset
from random import randrange
import itertools
import torch
import os
import pytest
from typing import Tuple
import pytorch_lightning as pl
from pytorch_lightning.trainer import supporters


FRAMES_PER_VIDEO = 8
EMBEDDING_DIM = 2048
USER = os.getenv("USER")


class VideoDataset(Dataset):
    """Generates a dummy video dataset.
    Returns: dataset with {"video": (batch_size, 3, frames, 224, 224)}
    """

    def __init__(
        self,
        size: Tuple[int] = (3, FRAMES_PER_VIDEO, 224, 224),
        num_samples: int = 252,
        num_classes: int = 400,
        clips_per_video: int = 4,
    ):
        """
        Args:
            size: tuple
            num_samples: number of samples
        """
        self.len = num_samples
        self.videos = torch.randn(num_samples, *size)
        self.labels = torch.randint(num_classes, (num_samples,))

        self.clips_per_video = clips_per_video
        self.clip_index = list(range(clips_per_video)) * num_samples
        num_videos = num_samples // clips_per_video
        self.video_index = [
            [randrange(10000)] * clips_per_video for _ in range(num_videos)
        ]
        self.video_index = list(itertools.chain(*self.video_index))

    def __getitem__(self, index):
        return {
            "video": self.videos[index],
            "label": self.labels[index],
            "clip_index": self.clip_index[index],
            "shot_index": self.clip_index[index],
            "video_index": self.video_index[index],
        }

    def __len__(self):
        return self.len



class SimCLRImageDataset(Dataset):
    """Generates a dummy dataset for SimCLR.
    Returns: dataset with x1, x2, x and y
    """

    def __init__(
        self,
        num_samples: int = 252,
        num_classes: int = 400,
    ):
        """
        Args:
            size: tuple
            num_samples: number of samples
        """
        self.len = num_samples
        self.labels = torch.randint(num_classes, (num_samples,))
        self.x1 = torch.rand(num_samples, 3, 224, 224)
        self.x2 = torch.rand(num_samples, 3, 224, 224)
        self.x = torch.rand(num_samples, 3, 224, 224)

    def __getitem__(self, index):
        return (
            (
                self.x1[index],
                self.x2[index],
                self.x[index],
            ),
            self.labels[index],
        )

    def __len__(self):
        return self.len


class SimCLRImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8):
        super().__init__()
        self.ds = SimCLRImageDataset()
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)


class ImageDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_classes: int = 15):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ds = DummyDataset((3, 224, 224), (self.num_classes,))

    def train_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)

    def val_dataloader(self):
        return DataLoader(self.ds, batch_size=self.batch_size, num_workers=1)


class ShapeNetImageDataset(Dataset):
    """Generates a dummy dataset for SimCLR.
    Returns: dataset with x1, x2, x and y
    """

    def __init__(
        self,
        num_samples: int = 252,
        num_classes: int = 15,
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
        return self.x[index], self.labels[index], self.fov

    def __len__(self):
        return self.len


class ShapeNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int = 8, num_classes: int = 15):
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.ds = ShapeNetImageDataset(num_classes=self.num_classes)

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
