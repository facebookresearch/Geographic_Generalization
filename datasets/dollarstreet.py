from torchvision import transforms
from PIL import Image
import os
import requests
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from ast import literal_eval
import pytorch_lightning as pl


class DollarstreetDataset(Dataset):

    timm_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ]
    )

    imagenet_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    def __init__(
        self,
        file_path: str = "/checkpoint/meganrichards/datasets/dollarstreet/interplay_version.csv",
        data_dir: str = "/checkpoint/marksibrahim/datasets/dollarstreet/",
        transform=timm_transform,
        return_type="image",
    ):
        self.file = pd.read_csv(file_path, index_col=0)
        self.file["label_1k"] = self.file["label_1k"].apply(literal_eval)
        self.data_dir = data_dir
        self.transform = transform
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

        if self.transform:
            image = self.transform(image)

        return image, label


class DollarStreetDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size: int = 8, num_classes: int = 55, num_workers=8, image_size=224
    ):
        """Dollarstreet Dataset

        Args:
            batch_size (int, optional): _description_. Defaults to 8.
            num_classes (int, optional): _description_. Defaults to 55.
        """
        super().__init__()
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.num_workers = num_workers
        self.ds = DollarstreetDataset()

        self.train_loader_names = ["train"]
        self.val_loader_names = ["val"]
        self.test_loader_names = ["test_1"]

    def val_dataloader(self):
        return DataLoader(
            self.ds, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.ds, batch_size=self.batch_size, num_workers=self.num_workers
        )
