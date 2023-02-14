from torchvision import transforms
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
import pytorch_lightning as pl
from datasets.image_datamodule import ImageDataModule
from typing import Callable
from torchvision import transforms
from PIL import Image
import os
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
        file_path: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/images_v2_with_imagenet_indices.csv",
        data_dir: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/",
        augmentations=timm_transform,
    ):
        self.file = pd.read_csv(file_path, index_col=0).reset_index()
        self.file["masked_imagenet_index"] = self.file["masked_imagenet_index"].apply(
            literal_eval
        )
        self.data_dir = data_dir
        self.augmentations = augmentations

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        row = self.file.iloc[idx]

        label = row["masked_imagenet_index_str"]

        image_name = row["imageRelPath"]
        image_path = os.path.join(self.data_dir, image_name)

        image = Image.open(image_path)

        if self.augmentations:
            image = self.augmentations(image)

        return image, label, image_name


class DollarStreetDataModule(ImageDataModule):
    def __init__(
        self,
        data_dir: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/",
        batch_size: int = 2,
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
        ds = DollarstreetDataset(augmentations=augmentations)
        self.file = ds.file
        return ds

    def _get_test_(self, path, augmentations):
        ds = DollarstreetDataset(augmentations=augmentations)
        self.file = ds.file
        return ds

    def test_transform(self) -> Callable:
        return timm_transform
