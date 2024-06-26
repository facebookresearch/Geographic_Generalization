"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
from datasets.image_datamodule import ImageDataModule
import numpy as np
from datasets.image_datamodule import IMAGENET_NORMALIZATION
from torchvision import transforms as transform_lib


class DollarstreetDataset(Dataset):
    def __init__(
        self,
        file_path: str = "data/dollarstreet/dataset_dollarstreet/house_separated_with_metadata/test.csv",
        data_dir: str = "data/dollarstreet/dataset_dollarstreet/",
        augmentations=transform_lib.Compose(
            [
                transform_lib.Resize(256),
                transform_lib.CenterCrop(224),
                transform_lib.ToTensor(),
                IMAGENET_NORMALIZATION,
            ]
        ),
        label_col="imagenet_sysnet_id",  # replace with 'topic_indicies' to get original dollarstreet label index instead of ImageNet
    ):
        self.file = pd.read_csv(file_path, index_col=0).reset_index()
        self.label_col = label_col
        self.file[label_col] = self.file[label_col].apply(literal_eval)

        if "imagenet" in label_col:
            print("Using 1k mapping for DollarStreet")
        else:
            print("Using DollarStreet original labels")

        self.data_dir = data_dir
        self.augmentations = augmentations

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        row = self.file.iloc[idx]
        label = ",".join(str(x) for x in row[self.label_col])
        image_name = row["imageRelPath"]
        image_path = os.path.join(self.data_dir, image_name)
        identifier = row["id"]

        image = Image.open(image_path)
        if np.array(image).shape[2] != 3:
            image = image.convert("RGB")

        if self.augmentations:
            image = self.augmentations(image)

        return image, label, identifier


class DollarStreetDataModule(ImageDataModule):
    def __init__(
        self,
        data_dir: str = "data/dollarstreet_kaggle/dataset_dollarstreet/",
        batch_size: int = 32,
        num_workers=8,
        image_size=224,
        label_col="imagenet_synset_id",
        house_separated_only=False,  # If set to True, this will use a randomly generated training/test split based on the household
        house_separated_and_region_balanced=False,  # If set to True, this will use a subsampled version of DollarStreet by region, such that regions has (not exactly) balanced subsets by region.
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )
        self.label_col = label_col
        self.file = {}
        self.house_separated_only = house_separated_only
        self.house_separated_and_region_balanced = house_separated_and_region_balanced
        if self.house_separated_only and self.house_separated_and_region_balanced:
            raise Exception(
                "Sorry - DollarStreet (Kaggle version) has mutually exclusive dataset split options. Please set one of 'house_separated_only' or 'house_separated_and_region_balanced' parameters to be False."
            )

    def get_path(self, stage):
        if self.house_separated_only:
            path = f"data/dollarstreet/dataset_dollarstreet/house_separated_with_metadata/{stage}.csv"
        elif self.house_separated_and_region_balanced:
            path = f"data/dollarstreet/dataset_dollarstreet/house_separated_region_balanced_with_metadata/{stage}.csv"
        else:
            path = f"data/dollarstreet/dataset_dollarstreet/splits_with_metadata/{stage}_with_groups.csv"

        print("Generating DS dataset with path ", path)
        return path

    def _get_dataset(self, path, stage, augmentations):
        ds = DollarstreetDataset(
            file_path=self.get_path(stage),
            label_col=self.label_col,
            augmentations=augmentations,
        )

        self.file[stage] = ds.file
        return ds


MAPPING = {
    "home": "manufactured home",
    "street view": "street sign",
    "tv": "television",
    "washing clothes/cleaning": "washing machine",
    "toilet": "toilet seat",
    "kitchen sink": "washbasin",
    "drinking water": "water bottle",
    "stove/hob": "stove",
    "salt": "salt shaker",
    "bed": "day bed",
    "toys": "toyshop",
    "everyday shoes": "running shoe",
    "plate of food": "plate",
    "cooking pots": "skillet",
    "social drink": "soda bottle",
    "phone": "cellphone",
    "place where eating dinner": "dining table",
    "lock on front door": "padlock",
    "wardrobe": "wardrobe",
    "soap for hands and body": "soap dispenser",
    "ceiling": "tile roof",
    "refrigerator": "refrigerator",
    "bathroom/toilet": "toilet seat",
    "dish washing brush/cloth": "dishrag",
    "toilet paper": "toilet paper",
    "plates": "plate",
    "dish washing soap": "soap dispenser",
    "trash/waste": "trash can",
    "dish racks": "plate rack",
    "shower": "shower curtain",
    "cups/mugs/glasses": "cup",
    "armchair": "rocking chair",
    "light sources": "table lamp",
    "light source in livingroom": "table lamp",
    "books": "bookcase",
    "switch on/off": "switch",
    "light source in kitchen": "table lamp",
    "couch": "studio couch",
    "sofa": "studio couch",
    "roof": "tile roof",
    "cutlery": "wooden spoon",
    "cooking utensils": "spatula",
    "medication": "medicine cabinet",
    "source of cool": "electric fan",
    "pen/pencils": "ballpoint",
    "street detail": "street sign",
    "turning lights on and off": "switch",
    "music equipment": "speaker",
    "tools": "tool kit",
    "cleaning equipment": "dishrag",
    "bed kids": "day bed",
    "table with food": "dining table",
    "get water": "water jug",
    "paper": "paper towel",
    "radio": "radio",
    "shoes": "running shoe",
    "starting stove": "igniter",
    "freezer": "icebox",
    "source of heat": "space heater",
    "computer": "desktop computer",
    "jewelry": "necklace",
    "knifes": "paper knife",
    "wall clock": "wall clock",
    "pouring water": "water jug",
    "doing dishes": "dishwasher",
    "guest bed": "day bed",
    "mosquito protection": "mosquito net",
    "bike": "all-terrain bike",
    "pouring drinking water": "water bottle",
    "oven": "stove",
    "place where serving guests": "eating place",
    "glasses or lenses": "dark glasses",
    "necklaces": "necklace",
    "source of light": "table lamp",
    "parking lot": "parking meter",
    "waste dumps": "trash can",
    "eating": "restaurant",
    "car": "passenger car",
    "reading light": "table lamp",
    "lightsources by bed": "table lamp",
    "family eating": "eating place",
    "arm watch": "digital watch",
    "taking a teaspoon of salt": "salt shaker",
    "using toilet": "toilet seat",
    "sitting and watching tv": "television",
    "opening and closing the freezer": "icebox",
    "diapers (or baby-pants)": "diaper",
    "moped/motorcycle": "moped",
    "cleaning after toilet": "toilet paper",
    "dishwasher": "dishwasher",
    "opening and closing the refrigerator": "refrigerator",
    "answering the phone": "mobile phone",
    "alarm clock": "analog clock",
    "wheel barrow": "wheelbarrow",
    "listening to the radio": "radio",
    "dinner guests": "eating place",
}
