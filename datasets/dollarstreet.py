from torchvision import transforms
from PIL import Image
import os
import requests
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
import pytorch_lightning as pl
from datasets.image_datamodule import ImageDataModule
from typing import Callable

timm_transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]
)


class DollarstreetDataset(Dataset):
    def __init__(
        self,
        file_path: str = "/checkpoint/meganrichards/datasets/dollarstreet/interplay_version_with_buckets.csv",
        data_dir: str = "/checkpoint/marksibrahim/datasets/dollarstreet/test/",
        augmentations=timm_transform,
        return_type="image",
    ):
        self.file = pd.read_csv(file_path, index_col=0)
        self.file["label_1k"] = self.file["label_1k"].apply(literal_eval)
        self.data_dir = data_dir
        self.augmentations = augmentations
        self.return_type = return_type

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        row = self.file.iloc[idx]
        url = row["url"]
        label = ", ".join(row["label_1k"])

        image_name = url.split("/")[-1]
        image_path = os.path.join(self.data_dir, image_name)

        if os.path.exists(image_path):
            image = Image.open(image_path)
        else:
            image = Image.open(requests.get(url, stream=True).raw)

        if self.augmentations:
            image = self.augmentations(image)

        return image, label, url


class DollarStreetDataModule(ImageDataModule):
    def __init__(
        self,
        data_dir: str = "/checkpoint/meganrichards/datasets/dollarstreet_interplay/",
        batch_size: int = 8,
        num_workers=8,
        image_size=224,
    ):
        """Dollarstreet Dataset

        Args:
            batch_size (int, optional): _description_. Defaults to 8.
            num_classes (int, optional): _description_. Defaults to 55.
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )

    def _get_dataset(self, path, augmentations):
        ds = DollarstreetDataset(data_dir=path, augmentations=augmentations)
        self.file = ds.file
        return ds

    def val_transform(self) -> Callable:
        return timm_transform
