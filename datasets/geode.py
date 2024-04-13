"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import pandas as pd

# import spacy
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from torch.utils.data import Dataset
from PIL import Image
import os
import pandas as pd
from ast import literal_eval
from datasets.image_datamodule import ImageDataModule
import numpy as np
from datasets.image_datamodule import IMAGENET_NORMALIZATION
from torchvision import transforms as transform_lib
from datasets.imagenet_classes import IMAGENET_CLASSES


class GeodeDataset(Dataset):
    def __init__(
        self,
        file_path: str = "data/geode/metadata_test_1k_final.csv",
        data_dir: str = "data/geode/images/",
        augmentations=transform_lib.Compose(
            [
                transform_lib.Resize(256),
                transform_lib.CenterCrop(224),
                transform_lib.ToTensor(),
                IMAGENET_NORMALIZATION,
            ]
        ),
        indices=[],
        label_col="1k_index",
    ):
        self.file = pd.read_csv(file_path, index_col=0).reset_index()
        if indices:
            self.file = self.file.iloc[indices]

        self.label_col = label_col
        if label_col == "1k_index":
            self.file[label_col] = self.file[label_col].apply(literal_eval)
            print("Using 1k mapping for GeoDE")
        elif label_col == "object_index":
            print("Using GeoDE original labels")
        else:
            raise Exception(
                "Geode has two options for the label_col parameter: 'object_index' for 0-40 geode object indexes, or '1k_index' for 1K indexes."
            )

        self.data_dir = data_dir
        self.augmentations = augmentations

    def __len__(self):
        return len(self.file)

    def __getitem__(self, idx):
        row = self.file.iloc[idx]
        if self.label_col == "1k_index":  # multilabel
            label = ",".join(str(x) for x in row[self.label_col])
        else:
            label = row[self.label_col]
        image_name = row["file_path"]
        image_path = os.path.join(self.data_dir, image_name)
        identifier = row["id"]  # idx

        image = Image.open(image_path)
        if np.array(image).shape[2] != 3:
            image = image.convert("RGB")

        if self.augmentations:
            image = self.augmentations(image)

        return image, label, identifier


class GeodeDataModule(ImageDataModule):
    def __init__(
        self,
        data_dir: str = "data/geode/images/",
        batch_size: int = 32,
        num_workers=8,
        image_size=224,
        indices=[],
        label_col="1k_index",
    ):
        """Geode Dataset

        Args:
            batch_size (int, optional): _description_. Defaults to 32.
        """
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )
        self.indices = indices
        self.label_col = label_col

    def _get_dataset(self, path, stage, augmentations):
        ds = GeodeDataset(
            file_path=f"data/geode/metadata_{stage}_1k_final.csv",
            augmentations=augmentations,
            indices=self.indices,
            label_col=self.label_col,
        )
        self.file = ds.file
        return ds


GEODE_CLASSES_TO_IMAGENET_CLASSES = {
    "bag": [
        "backpack",
        "purse",
        "punching bag",
        "sleeping bag",
        "plastic bag",
        "messenger bag",
        "shopping basket",
        "pencil case",
    ],
    "hand soap": ["soap dispenser", "lotion"],
    "dustbin": ["bucket", "trash can", "plastic bag", "barrel"],
    "toothbrush": [],
    "toothpaste toothpowder": [],
    "hairbrush comb": [],
    "chair": ["barber chair", "folding chair", "rocking chair", "couch", "throne"],
    "hat": [
        "cowboy hat",
        "swimming cap",
        "football helmet",
        "poke bonnet",
        "sombrero",
        "military hat (bearskin or shako)",
        "shower cap",
    ],
    "light fixture": ["table lamp", "spotlight", "lampshade", "candle"],
    "light switch": ["electrical switch"],
    "plate of food": ["plate", "tray"],
    "spices": [],
    "stove": ["Dutch oven", "stove"],
    "cooking pot": [
        "frying pan",
        "hot pot",
        "Crock Pot",
        "cauldron",
        "Dutch oven",
        "wok",
    ],
    "cleaning equipment": [
        "vacuum cleaner",
        "washing machine",
        "mop",
        "broom",
        "bucket",
        "soap dispenser",
    ],
    "lighter": ["lighter"],
    "medicine": ["pill bottle", "medicine cabinet"],
    "candle": ["candle"],
    "toy": ["teddy bear", "toy store"],
    "jug": ["water jug", "whiskey jug", "water bottle", "drink pitcher"],
    "streetlight lantern": ["torch", "pole"],
    "front door": ["sliding door"],
    "tree": [],
    "house": ["cliff dwelling", "mobile home", "barn", "home theater", "boathouse"],
    "backyard": ["patio"],
    "truck": ["garbage truck", "semi-trailer truck", "tow truck", "pickup truck"],
    "waste container": ["plastic bag", "trash can", "barrel", "bucket"],
    "car": [
        "garbage truck",
        "recreational vehicle",
        "semi-trailer truck",
        "tow truck",
        "sports car",
        "railroad car",
        "minivan",
        "station wagon",
        "minibus",
        "jeep",
        "limousine",
        "taxicab",
        "convertible",
        "pickup truck",
        "moving van",
        "police van",
        "race car",
    ],
    "fence": ["chain-link fence", "picket fence", "split-rail fence"],
    "road sign": ["traffic or street sign"],
    "dog": [
        "Bernese Mountain Dog",
        "Sealyham Terrier",
        "Toy Poodle",
        "toy terrier",
        "African wild dog",
        "husky",
        "Maltese",
        "Beagle",
        "Labrador Retriever",
        "Cairn Terrier",
        "dingo",
        "Australian Kelpie",
        "German Shepherd Dog",
        "Golden Retriever",
        "Malinois",
        "Norwegian Elkhound",
        "Chihuahua",
        "Tibetan Mastiff",
        "Staffordshire Bull Terrier",
        "American Staffordshire Terrier",
        "Pembroke Welsh Corgi",
        "Miniature Poodle",
        "Basenji",
        "Rhodesian Ridgeback",
        "Appenzeller Sennenhund",
        "Ibizan Hound",
    ],
    "wheelbarrow": ["wheelbarrow"],
    "religious building": ["mosque", "church", "monastery", "bell tower", "altar"],
    "stall": [],
    "boat": [
        "motorboat",
        "canoe",
        "fireboat",
        "lifeboat",
        "sailboat",
        "submarine",
        "ocean liner",
        "trimaran",
        "catamaran",
    ],
    "monument": [
        "triumphal arch",
        "obelisk",
        "stupa",
        "pedestal",
        "brass memorial plaque",
        "megalith",
    ],
    "flag": ["flagpole"],
    "bus": ["minibus", "school bus", "trolleybus"],
    "storefront": [
        "grocery store",
        "tobacco shop",
        "bookstore",
        "toy store",
        "barbershop",
        "candy store",
        "shoe store",
    ],
    "bicycle": ["tricycle", "mountain bike", "tandem bicycle", "unicycle"],
}

GEODE_CLASSES_TO_IMAGENET_INDICIES = {
    "bag": [414, 748, 747, 797, 728, 636, 790, 709],
    "hand soap": [804, 631],
    "dustbin": [463, 412, 728, 427],
    "toothbrush": [],
    "toothpaste toothpowder": [],
    "hairbrush comb": [],
    "chair": [423, 559, 765, 831, 857],
    "hat": [515, 433, 560, 452, 808, 439, 793],
    "light fixture": [846, 818, 619, 470],
    "light switch": [844],
    "plate of food": [923, 868],
    "spices": [],
    "stove": [544, 827],
    "cooking pot": [567, 926, 521, 469, 544, 909],
    "cleaning equipment": [882, 897, 840, 462, 463, 804],
    "lighter": [626],
    "medicine": [720, 648],
    "candle": [470],
    "toy": [850, 865],
    "jug": [899, 901, 898, 725],
    "streetlight lantern": [862, 733],
    "front door": [799],
    "tree": [],
    "house": [500, 660, 425, 598, 449],
    "backyard": [706],
    "truck": [569, 867, 864, 717],
    "waste container": [728, 412, 427, 463],
    "car": [
        569,
        757,
        867,
        864,
        817,
        705,
        656,
        436,
        654,
        609,
        627,
        468,
        511,
        717,
        675,
        734,
        751,
    ],
    "fence": [489, 716, 912],
    "road sign": [919],
    "dog": [
        239,
        190,
        265,
        158,
        275,
        248,
        153,
        162,
        208,
        192,
        273,
        227,
        235,
        207,
        225,
        174,
        151,
        244,
        179,
        180,
        263,
        266,
        253,
        159,
        240,
        173,
    ],
    "wheelbarrow": [428],
    "religious building": [668, 497, 663, 442, 406],
    "stall": [],
    "boat": [814, 472, 554, 625, 914, 833, 628, 871, 484],
    "monument": [873, 682, 832, 708, 458, 649],
    "flag": [557],
    "bus": [654, 779, 874],
    "storefront": [582, 860, 454, 865, 424, 509, 788],
    "bicycle": [870, 671, 444, 880],
}
