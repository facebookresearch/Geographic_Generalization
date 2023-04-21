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

NEW_GEODE_CLASSES_TO_IMAGENET_CLASSES = {
    "bag": ["backpack", "purse", "punching bag", "sleeping bag", "plastic bag"],
    "hand soap": ["soap dispenser", "lotion"],
    "dustbin": ["bucket"],
    "toothbrush": [],
    "toothpaste toothpowder": [],
    "hairbrush comb": ["hair clip"],
    "chair": ["barber chair", "folding chair", "rocking chair", "couch"],
    "hat": ["cowboy hat", "swimming cap", "football helmet"],
    "light fixture": ["table lamp"],
    "light switch": ["electrical switch"],
    "plate of food": ["meatloaf", "soup bowl", "plate"],
    "spices": [],
    "stove": ["Dutch oven", "stove"],
    "cooking pot": ["frying pan", "hot pot", "Crock Pot"],
    "cleaning equipment": ["vacuum cleaner", "washing machine"],
    "lighter": ["lighter"],
    "medicine": ["pill bottle", "medicine cabinet"],
    "candle": ["candle"],
    "toy": ["teddy bear"],
    "jug": ["water jug", "whiskey jug", "water bottle"],
    "streetlight lantern": ["lighthouse", "torch"],
    "front door": ["sliding door"],
    "tree": [],
    "house": ["cliff dwelling", "mobile home", "barn", "home theater"],
    "backyard": ["patio"],
    "truck": ["garbage truck", "semi-trailer truck", "tow truck"],
    "waste container": ["plastic bag"],
    "car": [
        "garbage truck",
        "recreational vehicle",
        "semi-trailer truck",
        "tow truck",
        "sports car",
        "railroad car",
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
    ],
    "wheelbarrow": ["wheelbarrow"],
    "religious building": ["mosque", "church"],
    "stall": ["toilet seat"],
    "boat": ["motorboat", "canoe"],
    "monument": ["triumphal arch", "obelisk", "stupa"],
    "flag": ["flagpole"],
    "bus": ["minibus", "school bus"],
    "storefront": ["traffic or street sign", "grocery store"],
    "bicycle": ["tricycle", "mountain bike"],
}

NEW_GEODE_CLASSES_TO_IMAGENET_INDICIES = {
    "bag": [414, 748, 747, 797, 728],
    "hand soap": [804, 631],
    "dustbin": [463],
    "toothbrush": [],
    "toothpaste toothpowder": [],
    "hairbrush comb": [584],
    "chair": [423, 559, 765, 831],
    "hat": [515, 433, 560],
    "light fixture": [846],
    "light switch": [844],
    "plate of food": [962, 809, 923],
    "spices": [],
    "stove": [544, 827],
    "cooking pot": [567, 926, 521],
    "cleaning equipment": [882, 897],
    "lighter": [626],
    "medicine": [720, 648],
    "candle": [470],
    "toy": [850],
    "jug": [899, 901, 898],
    "streetlight lantern": [437, 862],
    "front door": [799],
    "tree": [],
    "house": [500, 660, 425, 598],
    "backyard": [706],
    "truck": [569, 867, 864],
    "waste container": [728],
    "car": [569, 757, 867, 864, 817, 705],
    "fence": [489, 716, 912],
    "road sign": [919],
    "dog": [239, 190, 265, 158, 275, 248, 153, 162, 208, 192],
    "wheelbarrow": [428],
    "religious building": [668, 497],
    "stall": [861],
    "boat": [814, 472],
    "monument": [873, 682, 832],
    "flag": [557],
    "bus": [654, 779],
    "storefront": [919, 582],
    "bicycle": [870, 671],
}


# def calculate_geode_top_classes_and_indices(n):
#     nlp = spacy.load("en_core_web_lg")

#     imagenet_classes = (
#         pd.read_csv("/datasets01/imagenet_full_size/061417/labels.txt", header=None)
#         .rename(columns={0: "Synset_ID", 1: "Class_Name"})["Class_Name"]
#         .unique()
#         .tolist()
#     )
#     geode_classes = [
#         " ".join(x.split("_"))
#         for x in pd.read_csv("/checkpoint/meganrichards/datasets/geode/index.csv")[
#             "object"
#         ]
#         .unique()
#         .tolist()
#     ]

#     top_indices = {}
#     top_classes = {}
#     for geode_class in geode_classes:
#         g = nlp(geode_class)
#         sim_scores = []
#         for imagenet_class in imagenet_classes:
#             i = nlp(imagenet_class)
#             sim = i.similarity(g)
#             sim_scores.append(sim)
#         top_ind = sorted(range(len(sim_scores)), key=lambda i: sim_scores[i])[-n:]
#         top_indices[geode_class] = top_ind
#         top_classes[geode_class] = [imagenet_classes[i] for i in top_ind]

#     top_indexes = {}
#     for k, v in top_classes.items():
#         inds = []
#         for v_i in v:
#             inds.append(imagenet_classes.index(v_i))
#         top_indexes[k] = inds

#     return top_classes, top_indexes


# def get_country_to_region():
#     return {
#         "Angola": 0,
#         "Botswana": 0,
#         "Cameroon": 0,
#         "Egypt": 0,
#         "Ghana": 0,
#         "Nigeria": 0,
#         "South_Africa": 0,
#         "Argentina": 1,
#         "Brazil": 1,
#         "Colombia": 1,
#         "Mexico": 1,
#         "Uruguay": 1,
#         "Venezuela": 1,
#         "China": 2,
#         "Hong_Kong": 2,
#         "Japan": 2,
#         "South_Korea": 2,
#         "Taiwan": 2,
#         "Bulgaria": 3,
#         "France": 3,
#         "Greece": 3,
#         "Italy": 3,
#         "Poland": 3,
#         "Romania": 3,
#         "Switzerland": 3,
#         "United_Kingdom": 3,
#         "Cyprus": 3,
#         "Germany": 3,
#         "Ireland": 3,
#         "Netherlands": 3,
#         "Portugal": 3,
#         "Spain": 3,
#         "Ukraine": 3,
#         "Indonesia": 4,
#         "Malaysia": 4,
#         "Philippines": 4,
#         "Singapore": 4,
#         "Thailand": 4,
#         "Jordan": 5,
#         "Saudi_Arabia": 5,
#         "Turkey": 5,
#         "United_Arab_Emirates": 5,
#         "Yemen": 5,
#     }


# def get_reg_to_number():
#     return {
#         "Africa": 0,
#         "Americas": 1,
#         "EastAsia": 2,
#         "Europe": 3,
#         "SouthEastAsia": 4,
#         "WestAsia": 5,
#     }


# def prep_geode_pickle():
#     master_csv = pd.read_csv("/checkpoint/meganrichards/datasets/geode/index.csv")

#     image_names = []
#     obj = []
#     reg = []

#     country_to_region = get_country_to_region()  # maps country:number
#     region_to_number = get_reg_to_number()  # maps region name:region number
#     number_to_region = {
#         v: k for (k, v) in region_to_number.items()
#     }  # changes map to number: region

#     all_obj_names = sorted(
#         list(master_csv["object"].unique())
#     )  # gets the 38 categories in alpha order
#     print(all_obj_names)

#     for idx in master_csv.index:
#         fname = master_csv["file_path"][
#             idx
#         ]  # master_csv['file_name'][idx] # country_category_#.jpg
#         oname = master_csv["object"][
#             idx
#         ]  # master_csv['script_name'][idx] # object category of the image
#         cname = master_csv["ip_country"][idx].replace(" ", "_")  # country of the image
#         image_names.append(
#             ".../{}/{}/{}".format(
#                 number_to_region[country_to_region[cname]], cname, fname
#             )
#         )
#         obj.append(
#             all_obj_names.index(oname)
#         )  # number label corresponding to the object classification
#         reg.append(country_to_region[cname])  # appends region number of image

#         if idx % 1000 == 0:
#             print(idx)

#     (
#         train_names,
#         valtest_names,
#         train_obj,
#         valtest_obj,
#         train_reg,
#         valtest_reg,
#     ) = train_test_split(image_names, obj, reg, random_state=42, test_size=0.4)
#     val_names, test_names, val_obj, test_obj, val_reg, test_reg = train_test_split(
#         valtest_names, valtest_obj, valtest_reg, random_state=42, test_size=0.5
#     )

#     with open(
#         "/checkpoint/meganrichards/datasets/geode/geode_prep.pkl", "wb+"
#     ) as handle:
#         pickle.dump(
#             {
#                 "train": [train_names, train_obj, train_reg],
#                 "val": [val_names, val_obj, val_reg],
#                 "test": [test_names, test_obj, test_reg],
#             },
#             handle,
#         )

import pandas as pd
import pickle


def generate_metadata_csvs_from_pickle(save=True):
    file = open("/checkpoint/meganrichards/datasets/geode/geode_prep.pkl", "rb")
    data = pickle.load(file)
    meta = pd.read_csv("/checkpoint/meganrichards/datasets/geode/index.csv")

    ### Filter paths and save to CSVs
    direct_paths_train = [x.split(".../")[1] for x in data["train"][0]]
    direct_paths_simplified_train = [
        "/".join(x.split("/")[2:]) for x in direct_paths_train
    ]
    train_metadata = meta[meta["file_path"].isin(direct_paths_simplified_train)]
    assert len(train_metadata) == len(data["train"][0])

    direct_paths_val = [x.split(".../")[1] for x in data["val"][0]]
    direct_paths_simplified_val = ["/".join(x.split("/")[2:]) for x in direct_paths_val]
    val_metadata = meta[meta["file_path"].isin(direct_paths_simplified_val)]
    assert len(val_metadata) == len(data["val"][0])

    direct_paths_test = [x.split(".../")[1] for x in data["test"][0]]
    direct_paths_simplified_test = [
        "/".join(x.split("/")[2:]) for x in direct_paths_test
    ]
    test_metadata = meta[meta["file_path"].isin(direct_paths_simplified_test)]
    assert len(test_metadata) == len(data["test"][0])

    # Save metadata to files
    if save:
        train_metadata.to_csv(
            "/checkpoint/meganrichards/datasets/geode/metadata_train.csv"
        )
        val_metadata.to_csv("/checkpoint/meganrichards/datasets/geode/metadata_val.csv")
        test_metadata.to_csv(
            "/checkpoint/meganrichards/datasets/geode/metadata_test.csv"
        )
    return


def add_label_index(obj):
    object_labels_list = sorted(list(NEW_GEODE_CLASSES_TO_IMAGENET_INDICIES.keys()))
    return object_labels_list.index(obj.replace("_", " "))


def add_imagenet_index(obj):
    return NEW_GEODE_CLASSES_TO_IMAGENET_INDICIES[obj.replace("_", " ")]


def add_indexes_and_filter_by_1k(
    file_path="/checkpoint/meganrichards/datasets/geode/metadata_test.csv",
):
    metadata = pd.read_csv(file_path, index_col=0)

    # Add 1K index, and filter
    metadata["1k_index"] = metadata["object"].apply(add_imagenet_index)
    metadata_with_1k_mapping = metadata[
        metadata["1k_index"].apply(lambda x: len(x) > 0)
    ]

    # Add object index
    metadata_with_1k_mapping["object_index"] = metadata_with_1k_mapping["object"].apply(
        add_label_index
    )

    print("Original Number of Samples: ", len(metadata))
    print("Filtered Number of Samples: ", len(metadata_with_1k_mapping))

    return metadata_with_1k_mapping


def generate_1k_versions_of_metadata_csvs(save=False):
    for stage in ["train", "val", "test"]:
        print("\n-- Adding labels to", stage, "set --")
        file_path = f"/checkpoint/meganrichards/datasets/geode/metadata_{stage}.csv"
        metadata_1k_filtered = add_indexes_and_filter_by_1k(file_path)
        if save:
            metadata_1k_filtered.to_csv(
                f"/checkpoint/meganrichards/datasets/geode/metadata_{stage}_1k.csv"
            )


class GeodeDataset(Dataset):
    def __init__(
        self,
        file_path: str = "/checkpoint/meganrichards/datasets/geode/metadata_test_1k.csv",
        data_dir: str = "/checkpoint/meganrichards/datasets/geode/images/",
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
        identifier = idx

        image = Image.open(image_path)
        if np.array(image).shape[2] != 3:
            image = image.convert("RGB")

        if self.augmentations:
            image = self.augmentations(image)

        return image, label, identifier


class GeodeDataModule(ImageDataModule):
    def __init__(
        self,
        data_dir: str = "/checkpoint/meganrichards/datasets/geode/images/",
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
            file_path=f"/checkpoint/meganrichards/datasets/geode/metadata_{stage}_1k.csv",
            augmentations=augmentations,
            indices=self.indices,
            label_col=self.label_col,
        )
        self.file = ds.file
        return ds
