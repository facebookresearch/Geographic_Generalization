import os
import pytorch_lightning as pl
import torchvision.transforms as torch_transforms
from datasets.shapenet.shapenet_generation.attributes import Views
from datasets.shapenet.shapenet_generation.shapenet_synset_to_class_name import (
    SYNSET_TO_CLASS_NAME,
)
import torch
import json
import pandas as pd
import itertools
from typing import List, OrderedDict, Tuple, Any, Dict, Optional, Set
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.trainer import supporters
import matplotlib.pyplot as plt
from datasets.shapenet.augmentations import (
    DetRandomResizedCrop,
    DetRandomHorizontalFlip,
    DetRandomApply,
    DetGaussianBlur,
    DetColorJitter,
)

DET_TRANSFORMS = [
    DetRandomResizedCrop,
    DetRandomHorizontalFlip,
    DetRandomApply,
    DetGaussianBlur,
    DetColorJitter,
]
import random
from ast import literal_eval
import random
import numpy as np

USER = os.getenv("USER")
# computed across subset of training instances
MEANS = [0.1154, 0.1129, 0.1017]  # These are the 52 classes mean
STDS = [1.1017, 1.0993, 1.0832]


class ShapeNet(Dataset):
    """Dataset for fetch ShapeNet objects with specific poses.

    Args:
        data_dir: parent directory containing rendered images and fov.csv
        poses: list of tuple containing (x, y, z) pose angles
        instance_ids: list of instance ids to select for the dataset
        use_imagenet_classes: if true, synsets are mapped to their imagenet class index.
    """

    def __init__(
        self,
        data_dir: str = "/checkpoint/dianeb/shapenet_renderings",
        image_net_class_mappings: str = "/checkpoint/marksibrahim/datasets/imagenet_class_id_to_label.csv",
        instance_ids: List[str] = [
            "1101db09207b39c244f01fc4278d10c1",
            "11e925e3ea180b583388c2584b2f0f90",
        ],
        poses: List[Tuple[float, float, float]] = [(0.0, 0.0, 0.0)],
        use_imagenet_classes: bool = False,
        mean: List[float] = MEANS,
        std: List[float] = STDS,
        img_transforms: List[torch.nn.Module] = [
            torch_transforms.Resize(256),
            torch_transforms.CenterCrop(224),
        ],
        online_transforms: List[torch.nn.Module] = [
            torch_transforms.Resize(256),
            torch_transforms.CenterCrop(224),
        ],
    ):
        self.data_dir = data_dir
        self.image_net_class_mappings = image_net_class_mappings
        self.instance_ids = instance_ids
        self.poses = poses
        self.use_imagenet_classes = use_imagenet_classes

        self.mean = mean
        self.std = std

        self.imagenet_class_df = pd.read_csv(
            self.image_net_class_mappings,
            names=["class_id", "class_idx", "class_name"],
        )

        self.fov_df = self.load_fov(self.data_dir)
        indices = [
            (t[0], t[1][0], t[1][1], t[1][2])
            for t in itertools.product(instance_ids, poses)
        ]
        self.fov_filtered_df = self.fov_df.loc[indices]
        self.class_to_idx = self.map_class_to_idx()

        if use_imagenet_classes:
            self.class_to_idx = self.map_class_to_imagenet_idx()

        self.img_transforms = img_transforms
        self.online_transforms = online_transforms
        self.to_tensor = torch_transforms.Compose(
            self.img_transforms
            + [
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        self.to_tensor_online = torch_transforms.Compose(
            self.online_transforms
            + [
                torch_transforms.ToTensor(),
                torch_transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def get_sample(self, idx: int) -> Tuple[str, Tuple[float, float, float]]:
        """Selects instance_id and pose from the filtered dataframe

        Args:
            idx: index of sample to return from product(ids, poses)

        Returns: instance_id, pose and image path
        """
        try:
            pose = self.fov_filtered_df.iloc[idx].name[1:]
        except IndexError as e:
            print(
                f"failed to fetch pose for {idx=}. row is {self.fov_filtered_df.iloc[idx]}"
            )
            raise e
        image_path = self.fov_filtered_df.iloc[idx]["image_path"]
        instance_id = self.fov_filtered_df.iloc[idx].name[0]
        return instance_id, pose, image_path

    def synset_to_class_name(self, synset: str) -> str:
        """Maps a synset n20393 -> dog (human readable class name)"""
        return SYNSET_TO_CLASS_NAME[synset]

    def map_class_to_idx(self) -> Dict[str, int]:
        """One-hot encodes classes based on synsets."""
        classes = sorted(self.fov_df["class"].unique())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def map_class_to_imagenet_idx(self) -> Dict[str, int]:
        """One-hot encodes classes to their ImageNet on synsets."""
        classes = sorted(self.fov_df["class"].unique())
        df = self.imagenet_class_df
        class_to_idx = {
            cls_name: df[df["class_id"] == "n" + cls_name].class_idx.item()
            for cls_name in classes
        }
        return class_to_idx

    @staticmethod
    def load_fov(data_dir: str) -> pd.DataFrame:
        df = pd.read_csv(
            os.path.join(data_dir, "fov.csv"),
            delimiter="\t",
            dtype={
                "class": str,
                "instance_id": str,
                "image_path": str,
                "pose_x": float,
                "pose_y": float,
                "pose_z": float,
            },
        )
        # set columns as indices for fast filtering
        df = df.set_index(["instance_id", "pose_x", "pose_y", "pose_z"])
        # sorting speeds up lookups
        df = df.sort_index()
        return df

    def get_image_info(
        self, instance_id: str, pose: Tuple[float, float, float], attribute="image_path"
    ) -> Any:
        matches = self.fov_filtered_df.loc[[instance_id], pose[0], pose[1], pose[2]][
            attribute
        ].values
        if len(matches) > 1:
            raise ValueError(
                f"found multiple matching instances for pose {pose} and instance {instance_id}"
            )
        elif len(matches) == 0:
            raise ValueError(
                f"found no matching instances for pose {pose} and instance {instance_id}"
            )
        return matches[0]

    def __getitem__(self, index: int):
        instance_id, pose, image_path = self.get_sample(index)
        image = self.pil_loader(image_path)

        x = self.to_tensor(image)
        label = self.get_image_info(instance_id, pose, attribute="class")
        label_idx = self.class_to_idx[label]
        synset = "n" + label
        fov = {
            "pose": pose,
            "synset": synset,
            "class_name": self.synset_to_class_name(synset),
            "image_path": image_path,
        }
        return x, label_idx, fov

    def plot(self, x: torch.Tensor, fov: dict):
        print(fov)
        plt.imshow(torch_transforms.ToPILImage()(x), interpolation="bicubic")

    def plot_original(self, i: int):
        instance_id, pose = self.get_sample(i)
        image_path = self.get_image_info(instance_id, pose, attribute="image_path")
        image = self.pil_loader(image_path)
        plt.imshow(image, interpolation="bicubic")

    @staticmethod
    def pil_loader(path: str) -> Image.Image:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            img.convert("RGB")
        return img

    def __len__(self):
        return len(self.fov_filtered_df)


class ShapeNetDataModule(pl.LightningDataModule):
    """
    Creates training dataloaders where only a subset of instances undergo variation:
        - training dataloaders: X_canonical, X_varied
        - validation dataloaders: vary(X_canonical), held_out_canonical, held_out_varied

    Args:
        train_prop_to_vary: float indicating proportion of training instances to vary
        views: a Views object indicating the pose angles to vary
        use_imagenet_classes: if true, synsets are mapped to their imagenet class index.
    """

    def __init__(
        self,
        data_dir: str = "/checkpoint/marksibrahim/datasets/shapenet_renderings",
        batch_size: int = 8,
        num_workers: int = 8,
        train_prop_to_vary: float = 0.5,
        use_imagenet_classes: bool = False,
        trainval_ids_file: str = None,
        test_ids_file: str = None,
    ):
        super().__init__()

        self.data_dir = data_dir
        if trainval_ids_file is None:
            self.trainval_ids_file = os.path.join(self.data_dir, "trainval_ids.txt")
        else:
            self.trainval_ids_file = trainval_ids_file
        if test_ids_file is None:
            self.test_ids_file = os.path.join(self.data_dir, "test_ids.txt")
        else:
            self.test_ids_file = test_ids_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_prop_to_vary = train_prop_to_vary
        self.use_imagenet_classes = use_imagenet_classes

        self.fov_df = ShapeNet.load_fov(self.data_dir)
        self.attributes = self.load_attributes()
        self.views = self.create_views(self.attributes)
        self.num_classes = len(self.fov_df["class"].unique())

        self.train_loader_names = [
            "train_canonical",
            "train_canonical_repeated",
            "train_diverse_2d",
            "train_diverse_3d",
        ]
        self.val_loader_names = [
            "val_canonical",
            "val_diverse_2d",
            "val_diverse_3d",
        ]
        self.test_loader_names = [
            "test_canonical",
            "test_diverse_2d",
            "test_diverse_3d",
            "diverse_2d_train_canonical",
            "diverse_3d_train_canonical",
        ]

        self.train_augmentations = [
            torch_transforms.RandomResizedCrop(224),
            torch_transforms.RandomHorizontalFlip(),
        ]
        self.val_augmentations = [
            torch_transforms.Resize(256),
            torch_transforms.CenterCrop(224),
        ]

        # only prepare data from rank 0
        self.prepare_data_per_node = False

    def setup(self, stage: Optional[str] = None):
        """Partitions training set based on fov.csv and creates datasets
        Only runs on main process, not all GPUs.
        """
        synsets, instance_ids = self.get_instance_ids_and_synsets()
        synset_to_instance_ids = self.build_synset_to_instance_ids(
            synsets, instance_ids
        )

        trainval_ids = pd.read_csv(self.trainval_ids_file, sep="\n", header=None)[
            0
        ].tolist()
        test_ids = pd.read_csv(self.test_ids_file, sep="\n", header=None)[0].tolist()

        train_ids, val_ids = self._split_class_balanced(
            synset_to_instance_ids,
            ids_to_include=set(trainval_ids),
            prop=0.875,  # 7 times more train_ids than val_ids, just like before
        )

        test_ids = self._select_ids(
            synset_to_instance_ids, ids_to_include=set(test_ids)
        )

        print(
            "instance count - train:",
            len(train_ids),
            "val:",
            len(val_ids),
            "test:",
            len(test_ids),
        )
        if self.train_prop_to_vary > 0.0:
            train_canonical_ids, train_varied_ids = self._split_class_balanced(
                synset_to_instance_ids,
                ids_to_include=set(train_ids),
                prop=(1 - self.train_prop_to_vary),
            )
        else:
            train_canonical_ids, train_varied_ids = train_ids, []
        self.create_datasets(train_canonical_ids, train_varied_ids, val_ids, test_ids)

    def create_datasets(
        self,
        train_canonical_ids: List[str],
        train_diverse_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
    ):
        """Creates datasets for eval"""
        canonical_ids = {
            "train": train_canonical_ids,
            "val": val_ids,
            "test": test_ids,
        }
        diverse_ids = {"train": train_diverse_ids, "val": val_ids, "test": test_ids}

        for stage in ["train", "val", "test"]:
            stage_transforms = (
                self.train_augmentations if stage == "train" else self.val_augmentations
            )
            setattr(
                self,
                f"{stage}_canonical",
                ShapeNet(
                    data_dir=self.data_dir,
                    instance_ids=canonical_ids[stage],
                    poses=[self.views.canonical],
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=stage_transforms,
                ),
            )
            setattr(
                self,
                f"{stage}_diverse_2d",
                ShapeNet(
                    data_dir=self.data_dir,
                    instance_ids=diverse_ids[stage],
                    poses=self.views.generate_planar(),
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=stage_transforms,
                ),
            )
            setattr(
                self,
                f"{stage}_diverse_3d",
                ShapeNet(
                    data_dir=self.data_dir,
                    instance_ids=diverse_ids[stage],
                    poses=self.views.generate_3d(),
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=stage_transforms,
                ),
            )

        self.diverse_2d_train_canonical = ShapeNet(
            data_dir=self.data_dir,
            instance_ids=train_canonical_ids,
            poses=self.views.generate_planar(),
            use_imagenet_classes=self.use_imagenet_classes,
            img_transforms=self.val_augmentations,
        )

        self.diverse_3d_train_canonical = ShapeNet(
            data_dir=self.data_dir,
            instance_ids=train_canonical_ids,
            poses=self.views.generate_3d(),
            use_imagenet_classes=self.use_imagenet_classes,
            img_transforms=self.val_augmentations,
        )

    def compute_mean_std(self) -> Tuple[torch.Tensor, torch.Tensor]:
        x_sum = torch.tensor([0.0, 0.0, 0.0])
        x_squared_sum = torch.tensor([0.0, 0.0, 0.0])
        n = 0

        for x, _, _ in self.train_canonical:
            x_sum += x.mean(axis=[1, 2])
            x_squared_sum += (x**2).mean(axis=[1, 2])
            n += 1

        n = float(n)  # this is to get float division on n/n-1
        mean = x_sum / n
        var = (x_squared_sum / (n - 1)) - (mean**2) * n / (n - 1)
        std = torch.sqrt(var)
        return mean, std

    def get_instance_ids_and_synsets(self) -> Tuple[List[str], List[str]]:
        """Returns a list of synsets and their corresponding instance ids."""
        synsets = self.fov_df.loc[self.fov_df.index.unique(level=0), 0, 0, 0][
            "class"
        ].values
        instance_ids = self.fov_df.loc[
            self.fov_df.index.unique(level=0), 0, 0, 0
        ].index.get_level_values(0)
        return synsets, instance_ids

    def build_synset_to_instance_ids(
        self, synsets: List[str], instance_ids: List[str]
    ) -> Dict[str, List[str]]:
        """Returns a dictionary of synset -> [instance_ids]"""
        synset_to_ids = dict()

        for synset, i_id in zip(synsets, instance_ids):
            ids = synset_to_ids.get(synset, [])
            ids.append(i_id)
            synset_to_ids[synset] = ids

        return synset_to_ids

    def load_attributes(self) -> Dict:
        file = os.path.join(self.data_dir, "attributes.json")
        with open(file) as f:
            attributes = json.load(f)
        return attributes

    def create_views(self, attributes: Dict) -> Views:
        a = attributes
        views = Views(
            num_views=a["num_views"],
            order=a["order"],
        )
        return views

    def _split_class_balanced(
        self,
        class_to_instances: Dict[str, List[str]],
        ids_to_include: Optional[Set[str]] = None,
        prop: float = 0.5,
    ) -> Tuple[List[str], List[str]]:
        """Splits instance ids into two partitions grouped by class label.
        This ensures splits are balanced per class.

        Args:
            class_to_instances: synset str -> [instance_ids]
            ids_to_include: set of instance ids to include. All others are ignored
            prop: proportion of ids to use for partition1
        """
        partition_1, partition_2 = [], []
        for _, instance_ids in class_to_instances.items():
            filtered_ids = instance_ids
            if ids_to_include:
                filtered_ids = [i_id for i_id in instance_ids if i_id in ids_to_include]
            p1_size = round(prop * len(filtered_ids))
            partition_1.extend(filtered_ids[:p1_size])
            partition_2.extend(filtered_ids[p1_size:])

        if len(partition_1) == 0 or len(partition_2) == 0:
            raise ValueError(f"Not enough instances to partition")

        return partition_1, partition_2

    def _select_ids(
        self,
        class_to_instances: Dict[str, List[str]],
        ids_to_include: Optional[Set[str]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Filter instance ids according to a list of ids

        Args:
            class_to_instances: synset str -> [instance_ids]
            ids_to_include: set of instance ids to include. All others are ignored
        """
        ids = []
        for _, instance_ids in class_to_instances.items():
            filtered_ids = instance_ids
            if ids_to_include:
                filtered_ids = [i_id for i_id in instance_ids if i_id in ids_to_include]
                ids.extend(filtered_ids)
        return ids

    def _split(self, ids: List[str], prop: float = 0.5) -> Tuple[List[str], List[str]]:
        """Splits the given list into two partitions based on proportion given"""
        n = int(prop * len(ids))
        set1_ids, set2_ids = ids[:n], ids[n:]
        return set1_ids, set2_ids

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
            drop_last=False,
        )

    def check_number_samples(self, loaders: List[DataLoader]):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            print("Size", torch.distributed.get_world_size())
            world_size = torch.distributed.get_world_size()
            for loader in loaders:
                if len(loader.dataset) < world_size:
                    raise ValueError(
                        "Loader {loader} not enough samples to give one per GPU"
                    )

    def train_dataloader(self) -> supporters.CombinedLoader:
        loaders = {
            "train_canonical": self._make_loader(self.train_canonical, num_workers=1),
            # in order for canonical instance counts to match total 2d and 3d diverse poses
            "train_canonical_repeated": self._make_loader(
                self.train_canonical, num_workers=1
            ),
            "train_diverse_2d": self._make_loader(self.train_diverse_2d),
            "train_diverse_3d": self._make_loader(self.train_diverse_3d),
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
                ),
                (
                    "val_diverse_2d",
                    self._make_loader(
                        self.val_diverse_2d, shuffle=False, num_workers=1
                    ),
                ),
                (
                    "val_diverse_3d",
                    self._make_loader(
                        self.val_diverse_3d, shuffle=False, num_workers=1
                    ),
                ),
            ]
        )
        assert (
            list(loaders.keys()) == self.val_loader_names
        ), "val loader names don't match"

        return list(loaders.values())

    def test_dataloader(self, domain=None) -> List[DataLoader]:
        loaders = OrderedDict(
            [
                (
                    "test_canonical",
                    self._make_loader(
                        self.test_canonical, shuffle=False, num_workers=1
                    ),
                ),
                (
                    "test_diverse_2d",
                    self._make_loader(
                        self.test_diverse_2d, shuffle=False, num_workers=1
                    ),
                ),
                (
                    "test_diverse_3d",
                    self._make_loader(
                        self.test_diverse_3d, shuffle=False, num_workers=1
                    ),
                ),
                (
                    "diverse_2d_train_canonical",
                    self._make_loader(self.diverse_2d_train_canonical, shuffle=False),
                ),
                (
                    "diverse_3d_train_canonical",
                    self._make_loader(self.diverse_3d_train_canonical, shuffle=False),
                ),
            ]
        )
        assert (
            list(loaders.keys()) == self.test_loader_names
        ), "test loader names don't match"
        if not domain or domain == "all":
            return list(loaders.values())
        raise ValueError(f"domain {domain} not supported")


class ShapeNetNoAugsDataModule(ShapeNetDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_augmentations = self.val_augmentations


class ShapeNetCanonicalDataModule(ShapeNetDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_loader_names = [
            "train_canonical",
        ]
        self.val_loader_names = ["val_canonical", "val_diverse_2d", "val_diverse_3d"]
        self.test_loader_names = [
            "test_canonical",
            "test_diverse_2d",
            "test_diverse_3d",
            "diverse_2d_train_canonical",
            "diverse_3d_train_canonical",
        ]

    def setup(self, stage: Optional[str] = None):
        """Partitions training set based on fov.csv and creates datasets
        Only runs on main process, not all GPUs.
        """
        synsets, instance_ids = self.get_instance_ids_and_synsets()
        synset_to_instance_ids = self.build_synset_to_instance_ids(
            synsets, instance_ids
        )

        trainval_ids = pd.read_csv(self.trainval_ids_file, sep="\n", header=None)[
            0
        ].tolist()
        test_ids = pd.read_csv(self.test_ids_file, sep="\n", header=None)[0].tolist()

        train_ids, val_ids = self._split_class_balanced(
            synset_to_instance_ids,
            ids_to_include=set(trainval_ids),
            prop=0.875,  # 7 times more train_ids than val_ids, just like before
        )

        print(
            "instance count - train:",
            len(train_ids),
            "val:",
            len(val_ids),
            "test:",
            len(test_ids),
        )
        self.create_datasets(train_ids, val_ids, test_ids)

    def create_datasets(
        self,
        train_ids: List[str],
        val_ids: List[str],
        test_ids: List[str],
    ):
        """Creates datasets for eval"""
        canonical_ids = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }
        diverse_ids = {"val": val_ids, "test": test_ids}

        for stage in ["train", "val", "test"]:
            stage_transforms = (
                self.train_augmentations if stage == "train" else self.val_augmentations
            )
            setattr(
                self,
                f"{stage}_canonical",
                ShapeNet(
                    data_dir=self.data_dir,
                    instance_ids=canonical_ids[stage],
                    poses=[self.views.canonical],
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=stage_transforms,
                ),
            )
        for eval_stage in ["val", "test"]:
            setattr(
                self,
                f"{eval_stage}_diverse_2d",
                ShapeNet(
                    data_dir=self.data_dir,
                    instance_ids=diverse_ids[eval_stage],
                    poses=self.views.generate_planar(),
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=self.val_augmentations,
                ),
            )
            setattr(
                self,
                f"{eval_stage}_diverse_3d",
                ShapeNet(
                    data_dir=self.data_dir,
                    instance_ids=diverse_ids[eval_stage],
                    poses=self.views.generate_3d(),
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=self.val_augmentations,
                ),
            )

        self.diverse_2d_train_canonical = ShapeNet(
            data_dir=self.data_dir,
            instance_ids=train_ids,
            poses=self.views.generate_planar(),
            use_imagenet_classes=self.use_imagenet_classes,
            img_transforms=self.val_augmentations,
        )

        self.diverse_3d_train_canonical = ShapeNet(
            data_dir=self.data_dir,
            instance_ids=train_ids,
            poses=self.views.generate_3d(),
            use_imagenet_classes=self.use_imagenet_classes,
            img_transforms=self.val_augmentations,
        )

    def train_dataloader(self) -> supporters.CombinedLoader:
        loaders = {
            "train_canonical": self._make_loader(self.train_canonical),
        }
        # max_size_cycle cycles through canonical poses to match size of diverse poses
        combined_loaders = supporters.CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def val_dataloader(self) -> List[DataLoader]:
        # use an ordereddict to ensure order is preserved
        loaders = OrderedDict(
            [
                ("val_canonical", self._make_loader(self.val_canonical, shuffle=False)),
                (
                    "val_diverse_2d",
                    self._make_loader(self.val_diverse_2d, shuffle=False),
                ),
                (
                    "val_diverse_3d",
                    self._make_loader(self.val_diverse_3d, shuffle=False),
                ),
            ]
        )
        assert (
            list(loaders.keys()) == self.val_loader_names
        ), "val loader names don't match"

        return list(loaders.values())


if __name__ == "__main__":
    pass
