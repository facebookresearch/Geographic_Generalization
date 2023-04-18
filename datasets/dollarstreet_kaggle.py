from PIL import Image
import os
import pandas as pd
from torch.utils.data import Dataset
from ast import literal_eval
from datasets.image_datamodule import ImageDataModule
import numpy as np
from datasets.image_datamodule import IMAGENET_NORMALIZATION


class DollarstreetDataset(Dataset):
    def __init__(
        self,
        file_path: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/images_v2_imagenet_test_with_income_and_region_groups_with_indices.csv",
        data_dir: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/",
        augmentations=IMAGENET_NORMALIZATION,
        label_col="imagenet_sysnet_id",  # topic_indicies
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
        print(augmentations)

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
        data_dir: str = "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/",
        batch_size: int = 32,
        num_workers=8,
        image_size=224,
        label_col="imagenet_synset_id",
    ):
        """Dollarstreet Dataset

        Args:
            batch_size (int, optional): _description_. Defaults to 32.
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )
        self.label_col = label_col

    def _get_dataset(self, path, augmentations):
        ds = DollarstreetDataset(label_col=self.label_col, augmentations=augmentations)
        self.file = ds.file
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


# def make_imagenet_class_to_dollarstreet_idx():
#     imagenet_class_to_dollarstreet_idx = {}

#     for m in list(MAPPING.values()):
#         for i, imagenet_class in IMAGENET1K_IDX_TO_NAMES.items():
#             if m.strip() in imagenet_class:
#                 if m.strip() in imagenet_class_to_dollarstreet_idx:
#                     imagenet_class_to_dollarstreet_idx[m.strip()].append(i)
#                 else:
#                     imagenet_class_to_dollarstreet_idx[m.strip()] = [i]

#     for cl, ls in imagenet_class_to_dollarstreet_idx.items():
#         imagenet_class_to_dollarstreet_idx[cl] = list(set(ls))
#     return imagenet_class_to_dollarstreet_idx


# def make_idx_to_label(imagenet_class_to_dollarstreet_idx):
#     idx_to_label = {}
#     for name, idx_list in imagenet_class_to_dollarstreet_idx.items():
#         for idx in idx_list:
#             if idx in idx_to_label:
#                 idx_to_label[idx].append(name)
#             else:
#                 idx_to_label[idx] = [name]

#     return idx_to_label


# def make_dollarstreet_mask():
#     imagenet_class_to_dollarstreet_idx = make_imagenet_class_to_dollarstreet_idx()
#     idx_to_label = make_idx_to_label(imagenet_class_to_dollarstreet_idx)
#     imagenet_mask = list(idx_to_label.keys())
#     return imagenet_mask
