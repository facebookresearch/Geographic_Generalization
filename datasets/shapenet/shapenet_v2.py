"""
Data is split into train, val, test once
X_train, X_test, y_train, y_test = cross_validation.train_test_split(Data, Target, test_size=0.3, random_state=0, stratify=Target)
"""


from pytorch_lightning.trainer import supporters
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as torch_transforms
from typing import List, Optional
import os
import pandas as pd
from PIL import Image
import random
from datasets.shapenet.shapenet import MEANS, STDS
import torch
import pytorch_lightning as pl
from collections import OrderedDict
import torchvision.transforms.functional as F
import torchvision
import matplotlib.pyplot as plt


class ShapeNetSampleVariationPerInstance(Dataset):
    """Shapes dataset with diverse sampling per image.
    For a given instance, we sample with train_prop_to_vary probability
        a diverse image for the specified factor.

    Dataset description:
    - 2700 instances
    - 54 classes
    - 100 parameters per factor, except Background which has 24

    Args:
        all_fov_df:  dataframe containing values for fov_idc.csv.
            Optional, if not provided it's read from the filesystem.
    """

    FACTORS = ["Rotation", "Translation", "Scale", "Spot hue", "Background path"]
    CANONICAL_VALUES = {
        "Translation": 50,
        "Rotation": 50,
        "Scale": 100,
        "Spot hue": 10,
        "Background path": "sky/1.jpg",
    }

    def __init__(
        self,
        data_dir: str = "/checkpoint/garridoq/datasets_fov/individual",
        train_prop_to_vary: float = 0.5,
        factor_to_vary: str = "Rotation",
        all_fov_df: Optional[pd.DataFrame] = None,
        instance_ids: List[str] = [
            "10155655850468db78d106ce0a280f87",
            "1021a0914a7207aff927ed529ad90a11",
            "100c3076c74ee1874eb766e5a46fceab",
            "1011e1c9812b84d2a9ed7bb5b55809f8",
            "101354f9d8dede686f7b08d9de913afe",
        ],
        img_transforms: List[torch.nn.Module] = [
            torch_transforms.Resize(256),
            torch_transforms.CenterCrop(224),
        ],
    ):
        self.data_dir = data_dir
        self.train_prop_to_vary = train_prop_to_vary
        self.factor_to_vary = factor_to_vary
        self.instance_ids = instance_ids
        self.img_transforms = img_transforms

        self.synset_to_name = self.read_synset_to_name()
        self.synset_to_class_idx = self.create_synset_to_class_idx()

        self.all_df = (
            self.make_data_df(self.data_dir) if all_fov_df is None else all_fov_df
        )
        self.df = self.all_df.loc[self.instance_ids]
        self.factor_df = self.select_varying_factor_rows(self.df)

        self.means, self.stds = MEANS, STDS

        self.to_tensor = torch_transforms.Compose(
            self.img_transforms
            + [
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(mean=self.means, std=self.stds),
            ]
        )

    def read_synset_to_name(self):
        df = pd.read_csv(
            "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/synset_to_name.csv",
            dtype={"Synset ID": str},
        )
        df = df.rename(columns={"Synset ID": "synset_id", "Synset Name": "class_name"})
        df = df.set_index("synset_id")
        df = df.sort_index()
        return df

    def create_synset_to_class_idx(self) -> dict:
        return {s: i for i, s in enumerate(sorted(self.synset_to_name.index.values))}

    def select_varying_factor_rows(self, df: pd.DataFrame):
        factor_df = df.copy()
        canonical_factors = [f for f in self.FACTORS if f != self.factor_to_vary]

        for canonical_factor in canonical_factors:
            factor_df = factor_df[factor_df[f"is_canonical_{canonical_factor}"]]
        return factor_df

    @classmethod
    def make_data_df(cls, data_dir: str) -> pd.DataFrame:
        df = pd.read_csv(os.path.join(data_dir, "fov_idcs.csv"), dtype={"Synset": str})
        # drop duplicates since canonical entries are repeated
        cols_to_consider_for_duplicates = df.columns.tolist()
        cols_to_consider_for_duplicates.remove("Image path")
        df = df.drop_duplicates(subset=cols_to_consider_for_duplicates)

        canonical_translation = (
            (df["Translation X"] == cls.CANONICAL_VALUES["Translation"])
            & (df["Translation Y"] == cls.CANONICAL_VALUES["Translation"])
            & (df["Translation Z"] == cls.CANONICAL_VALUES["Translation"])
        )
        df["is_canonical_Translation"] = canonical_translation
        canonical_rotation = (
            (df["Rotation X"] == cls.CANONICAL_VALUES["Rotation"])
            & (df["Rotation Y"] == cls.CANONICAL_VALUES["Rotation"])
            & (df["Rotation Z"] == cls.CANONICAL_VALUES["Rotation"])
        )
        df["is_canonical_Rotation"] = canonical_rotation
        canonical_scale = df["Scale"] == cls.CANONICAL_VALUES["Scale"]
        df["is_canonical_Scale"] = canonical_scale
        canonical_spot_hue = df["Spot hue"] == cls.CANONICAL_VALUES["Spot hue"]
        df["is_canonical_Spot hue"] = canonical_spot_hue
        canonical_background_path = df["Background path"].str.contains(
            cls.CANONICAL_VALUES["Background path"]
        )
        df["is_canonical_Background path"] = canonical_background_path

        df["is_canonical"] = (
            (canonical_translation)
            & (canonical_rotation)
            & (canonical_scale)
            & (canonical_spot_hue)
            & canonical_background_path
        )
        df = df.rename(columns={"Object": "instance_id"})
        df = df.set_index("instance_id")
        return df

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
        return img

    def draw_diverse(self) -> bool:
        """Returns a bool indicating whether to draw a diverse or canonical image"""
        if self.train_prop_to_vary == 0:
            return False
        draw = random.random()
        if draw < self.train_prop_to_vary:
            return True
        return False

    def get_df_entry(self, instance_id: str, canonical: bool) -> pd.DataFrame:
        if canonical:
            mask = self.factor_df.loc[instance_id]["is_canonical"]
            entry = self.factor_df.loc[instance_id][mask]
        else:
            mask = ~self.factor_df.loc[instance_id]["is_canonical"]
            entry = self.factor_df.loc[instance_id][mask].sample()
        return entry

    def __getitem__(self, idx: int):
        instance_id = self.instance_ids[idx]
        is_canonical = not self.draw_diverse()
        entry = self.get_df_entry(instance_id, is_canonical)
        synset = entry.Synset.item()
        class_name = self.synset_to_name.loc[synset].item()
        image_path = entry["Image path"].item()
        image = self.pil_loader(image_path)
        x = self.to_tensor(image)
        y = self.synset_to_class_idx[synset]

        fov = {
            "image_path": image_path,
            "synset": synset,
            "class_name": class_name,
            "instance_id": instance_id,
            "Translation": entry["Translation X"].item(),
            "Scale": entry["Scale"].item(),
            "Background path": entry["Background path"].item(),
            "Spot hue": entry["Spot hue"].item(),
            "Rotation": entry["Rotation X"].item(),
            "is_canonical": entry["is_canonical"].item(),
        }

        return x, y, fov

    def __len__(self):
        return len(self.instance_ids)


class ShapeNetAllDiverse(ShapeNetSampleVariationPerInstance):
    def __getitem__(self, idx: int):
        entry = self.factor_df.iloc[idx]
        instance_id = entry.name
        synset = entry.Synset
        class_name = self.synset_to_name.loc[synset].item()
        image_path = entry["Image path"]
        image = self.pil_loader(image_path)
        x = self.to_tensor(image)
        y = self.synset_to_class_idx[synset]

        fov = {
            "image_path": image_path,
            "synset": synset,
            "class_name": class_name,
            "instance_id": instance_id,
            "Translation": entry["Translation X"].item(),
            "Scale": entry["Scale"].item(),
            "Background path": entry["Background path"],
            "Spot hue": entry["Spot hue"].item(),
            "Rotation": entry["Rotation X"].item(),
            "is_canonical": entry["is_canonical"].item(),
        }

        return x, y, fov

    def __len__(self):
        return len(self.factor_df)


class ShapeNetAllCanonical(ShapeNetSampleVariationPerInstance):
    def draw_diverse(self) -> bool:
        return False


class ShapeNetv2SingleFactorDataModule(pl.LightningDataModule):
    train_augmentations = [
        torch_transforms.RandomResizedCrop(224),
        torch_transforms.RandomHorizontalFlip(),
    ]
    val_augmentations = [
        torch_transforms.Resize(256),
        torch_transforms.CenterCrop(224),
    ]

    def __init__(
        self,
        data_dir: str = "/checkpoint/garridoq/datasets_fov/individual",
        train_prop_to_vary: float = 0.5,
        factor_to_vary: str = "Rotation",
        batch_size: int = 8,
        num_workers: int = 4,
        train_ids_file: str = "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/train_ids_image_sampling.csv",
        val_ids_file: str = "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/val_ids_image_sampling.csv",
        test_ids_file: str = "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/test_ids_image_sampling.csv",
    ):
        super().__init__()

        self.data_dir = data_dir
        self.train_prop_to_vary = train_prop_to_vary
        self.factor_to_vary = factor_to_vary
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_ids_file = train_ids_file
        self.val_ids_file = val_ids_file
        self.test_ids_file = test_ids_file

        self.train_ids, self.val_ids, self.test_ids = self.get_instance_id_splits()
        self.num_classes = 54

        # only prepare data from rank 0
        self.prepare_data_per_node = False

        self.train_loader_names = ["train"]
        self.val_loader_names = ["val_canonical"] + [
            f"val_diverse_{fov}" for fov in ShapeNetSampleVariationPerInstance.FACTORS
        ]
        self.test_loader_names = ["test_canonical"] + [
            f"test_diverse_{fov}" for fov in ShapeNetSampleVariationPerInstance.FACTORS
        ]

        # loaded during setup()
        self.all_fov_df = None

    def get_instance_id_splits(self):
        train_ids = pd.read_csv(self.train_ids_file, header=None)[0].values.tolist()
        val_ids = pd.read_csv(self.val_ids_file, header=None)[0].values.tolist()
        test_ids = pd.read_csv(self.test_ids_file, header=None)[0].values.tolist()
        return train_ids, val_ids, test_ids

    def setup(self, stage: Optional[str] = None):
        self.all_fov_df = ShapeNetSampleVariationPerInstance.make_data_df(self.data_dir)
        self.create_datasets()

    def create_datasets(self):
        self.train_dataset = ShapeNetSampleVariationPerInstance(
            data_dir=self.data_dir,
            train_prop_to_vary=self.train_prop_to_vary,
            factor_to_vary=self.factor_to_vary,
            all_fov_df=self.all_fov_df,
            instance_ids=self.train_ids,
            img_transforms=self.train_augmentations,
        )
        self.val_canonical = ShapeNetAllCanonical(
            data_dir=self.data_dir,
            all_fov_df=self.all_fov_df,
            instance_ids=self.val_ids,
            img_transforms=self.val_augmentations,
        )
        self.test_canonical = ShapeNetAllCanonical(
            data_dir=self.data_dir,
            all_fov_df=self.all_fov_df,
            instance_ids=self.val_ids,
            img_transforms=self.val_augmentations,
        )

        for factor in ShapeNetSampleVariationPerInstance.FACTORS:
            setattr(
                self,
                f"val_diverse_{factor}",
                ShapeNetAllDiverse(
                    data_dir=self.data_dir,
                    factor_to_vary=factor,
                    all_fov_df=self.all_fov_df,
                    instance_ids=self.val_ids,
                    img_transforms=self.val_augmentations,
                ),
            )
            setattr(
                self,
                f"test_diverse_{factor}",
                ShapeNetAllDiverse(
                    data_dir=self.data_dir,
                    factor_to_vary=factor,
                    all_fov_df=self.all_fov_df,
                    instance_ids=self.test_ids,
                    img_transforms=self.val_augmentations,
                ),
            )

    def _make_loader(
        self, dataset: Dataset, shuffle: bool = True, num_workers=None
    ) -> DataLoader:
        if num_workers is None:
            # use default if not provided
            num_workers = self.num_workers
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            pin_memory=True,
        )

    def train_dataloader(self) -> supporters.CombinedLoader:
        loaders = {
            "train": self._make_loader(self.train_dataset),
        }
        # max_size_cycle cycles through canonical poses to match size of diverse poses
        combined_loaders = supporters.CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def val_dataloader(self) -> List[DataLoader]:
        # use an ordereddict to ensure order is preserved
        loaders = OrderedDict(
            [
                (
                    "val_canonical",
                    self._make_loader(self.val_canonical, shuffle=False, num_workers=1),
                )
            ]
            + [
                (
                    f"val_diverse_{fov}",
                    self._make_loader(
                        getattr(self, f"val_diverse_{fov}"),
                        shuffle=False,
                    ),
                )
                for fov in ShapeNetSampleVariationPerInstance.FACTORS
            ]
        )
        assert (
            list(loaders.keys()) == self.val_loader_names
        ), "val loader names don't match"

        return list(loaders.values())

    def test_dataloader(self) -> List[DataLoader]:
        loaders = OrderedDict(
            [
                (
                    "test_canonical",
                    self._make_loader(
                        self.test_canonical, shuffle=False, num_workers=1
                    ),
                )
            ]
            + [
                (
                    f"test_diverse_{fov}",
                    self._make_loader(
                        getattr(self, f"test_diverse_{fov}"),
                        shuffle=False,
                    ),
                )
                for fov in ShapeNetSampleVariationPerInstance.FACTORS
            ]
        )
        assert (
            list(loaders.keys()) == self.test_loader_names
        ), "test loader names don't match"

        return list(loaders.values())

    def show_batch(x) -> plt.figure:
        def show(imgs):
            if not isinstance(imgs, list):
                imgs = [imgs]

            fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
            for i, img in enumerate(imgs):
                img = img.detach()
                img = F.to_pil_image(img)
                axs[0, i].imshow(np.asarray(img))
                axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            return fig

        return show(torchvision.utils.make_grid(x))
