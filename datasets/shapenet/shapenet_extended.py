import os
import pytorch_lightning as pl
import torchvision.transforms as torch_transforms
import torch
import pandas as pd
import itertools
from typing import List, OrderedDict, Tuple, Any, Dict, Optional, Set
from torch.utils.data import DataLoader
from pytorch_lightning.trainer import supporters
import random
from datasets.shapenet.shapenet import ShapeNet, MEANS, STDS, ShapeNetDataModule
import numpy as np

CANONICAL_VALUES = {
    "Translation": 50,
    "Rotation": 50,
    "Scale": 100,
    "Spot hue": 10,
    "Background path": "/checkpoint/garridoq/datasets_fov/backgrounds/sky/1.jpg",
}
FOV_NAMES = sorted(
    list(CANONICAL_VALUES.keys())
)  # ORDER USED TO INDEX THE DATAFRAME AND TO CREATE FOV_INDICES. IMPORTANT TO FOLLOW THAT ORDER !!!

TEST_IDS_PATH = "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/test_ids.txt"
TRAIN_VAL_IDS_PATH = (
    "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/trainval_ids.txt"
)
TEST_IDS_SMALL_PATH = (
    "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/test_ids_small.txt"
)
TRAIN_VAL_IDS_SMALL_PATH = (
    "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/trainval_ids_small.txt"
)

BACKGROUND_PATHS = [
    "/checkpoint/garridoq/datasets_fov/backgrounds/sky/0.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/sky/1.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/sky/2.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/sky/3.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/sky/4.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/home/0.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/home/1.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/home/2.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/home/3.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/home/4.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/grass/0.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/grass/1.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/grass/2.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/grass/3.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/grass/4.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/city/0.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/city/1.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/city/2.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/city/3.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/city/4.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/water/0.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/water/1.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/water/2.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/water/3.jpg",
    "/checkpoint/garridoq/datasets_fov/backgrounds/water/4.jpg",
]


def merge_rotation_translation_single(original_df):
    """Function to merge the 3 columns of rotation and translation under a SINGLE SCALAR (assumed same for everybody) so we take X axis value"""
    translation_df = original_df[["Translation X"]]
    translation_df = translation_df.rename(columns={"Translation X": "Translation"})
    rotation_df = original_df[["Rotation X"]]
    rotation_df = rotation_df.rename(columns={"Rotation X": "Rotation"})
    df = original_df.drop(
        columns=[
            "Translation X",
            "Translation Y",
            "Translation Z",
            "Rotation X",
            "Rotation Y",
            "Rotation Z",
        ]
    )
    df = df.join(translation_df)
    df = df.join(rotation_df)
    return df


def merge_rotation_translation_tuple(original_df):
    """Function to merge the 3 columns of rotation and translation under one TUPLE"""
    translation_df = pd.DataFrame(
        original_df[["Translation X", "Translation Y", "Translation Y"]].apply(
            tuple, axis=1
        ),
        columns=["Translation"],
    )
    rotation_df = pd.DataFrame(
        original_df[["Rotation X", "Rotation Y", "Rotation Y"]].apply(tuple, axis=1),
        columns=["Rotation"],
    )
    df = original_df.drop(
        columns=[
            "Translation X",
            "Translation Y",
            "Translation Z",
            "Rotation X",
            "Rotation Y",
            "Rotation Z",
        ]
    )
    df = df.join(translation_df)
    df = df.join(rotation_df)
    return df


class AbstractFovHandler:
    def __init__(self, fov_dict: Dict = {"Rotation X": [0]}):
        self.fov_dict = fov_dict
        self.set_fov_indices()

    def __parse_dict__(self):

        self.varying_fov_names = sorted(list(self.fov_dict.keys()))
        fov_values_list = []
        for fov_name in FOV_NAMES:
            if fov_name in self.varying_fov_names:
                fov_values_list.append(self.fov_dict[fov_name])
            else:
                fov_values_list.append([CANONICAL_VALUES[fov_name]])

        self.fov_values_list = fov_values_list

    def set_fov_indices(self):
        self.__parse_dict__()
        # By default, will be the factorised support of the varying FoV range and non-varying in their canonical pose
        self.fov_indices = [t for t in itertools.product(*self.fov_values_list)]

    def set_fov_df(self, fov_df):
        self.fov_df = fov_df

    def return_filtered_indices(self, ids):

        indices = [(t[0],) + t[1] for t in itertools.product(ids, self.fov_indices)]
        return indices

    def __len__(self):

        return len(self.fov_indices)


class SingleFovHandler(AbstractFovHandler):

    """FoV Handler when we move 1 FoV. The rest of the FoV in the dict should be fixed to 1 value"""

    def __init__(self, fov_dict, fov: str = "Rotation X"):
        super().__init__(fov_dict)
        self.fov = fov


class PairFovHandler(AbstractFovHandler):

    """FoV Handler when we move 2 FoV in Pairs. As of now, we take all possible combinations"""

    def __init__(self, fov_dict, fov_1: str = "Rotation X", fov_2: str = "Rotation Y"):
        super().__init__(fov_dict)
        self.fov_1 = fov_1
        self.fov_2 = fov_2


class AllFovHandler(AbstractFovHandler):

    """FoV Handler when we move All FoV. As of now, we take a random subset of all possible combinations"""

    def __init__(self, fov_dict, n_indices=10, fov_indices_file=None):
        self.n_indices = n_indices
        self.draw_random = False
        self.fov_indices_file = fov_indices_file
        super().__init__(fov_dict)

    def set_fov_indices(self):

        if self.draw_random:
            self.parse_dict()
            # Select a random subset of the factorised support set, without replacement
            all_indices = [t for t in itertools.product(*self.fov_values_list)]
            self.fov_indices = random.sample(all_indices, self.n_indices)
        else:
            self.fov_values_list = None
            if self.fov_indices_file is None:
                raise ValueError(
                    "fov indices file must not be None if draw_random is False"
                )
            # Combinations passed externally
            indices = np.load(self.fov_indices_file)
            df = pd.DataFrame(
                indices,
                columns=[
                    "Translation X",
                    "Translation Y",
                    "Translation Z",
                    "Rotation X",
                    "Rotation Y",
                    "Rotation Z",
                    "Scale",
                    "Spot hue",
                    "Background path",
                ],
            )
            # We NEED THE INDICES TO FOLLOW THE FOV NAMES ORDER so that we correctly select them afterwards.
            # Problem is here the fov still have X,Y,Z information
            # TODO currently ugly way is to manually follow the order of FOV NAMES after ditching the extra Y Z axes
            df = merge_rotation_translation_single(df)
            df = df[FOV_NAMES]
            df = df.astype(
                {
                    "Translation": float,
                    "Rotation": float,
                    "Scale": float,
                    "Spot hue": float,
                },
                errors="raise",
            )
            self.fov_indices = [tuple(t) for t in df.values]
            self.varying_fov_names = sorted(
                list(CANONICAL_VALUES.keys())
            )  # all of them are varying


FOV_DIVERSE_HANDLERS = {
    "Translation": SingleFovHandler({"Translation": np.arange(0, 101, 1)}),
    "Rotation": SingleFovHandler({"Rotation": np.arange(0, 101, 1)}),
    "Scale": SingleFovHandler({"Scale": np.arange(0, 101, 1)}),
    "Spot hue": SingleFovHandler({"Spot hue": np.arange(0, 101, 1)}),
    "Background path": SingleFovHandler({"Background path": BACKGROUND_PATHS}),
}


class ShapeNetExtended(ShapeNet):
    """Dataset for fetch ShapeNet objects with multiple varying FoV."""

    def __init__(
        self,
        data_dir: str,
        fov_handler: AbstractFovHandler,
        image_net_class_mappings: str = "/checkpoint/marksibrahim/datasets/imagenet_class_id_to_label.csv",
        instance_ids: List[str] = [
            "10155655850468db78d106ce0a280f87",
        ],
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
        self.use_imagenet_classes = use_imagenet_classes
        self.mean = mean
        self.std = std

        self.imagenet_class_df = pd.read_csv(
            self.image_net_class_mappings,
            names=["class_id", "class_idx", "class_name"],
        )

        self.fov_handler = fov_handler
        fov_df = self.load_fov(self.data_dir)
        self.fov_handler.set_fov_df(fov_df)

        indices = self.fov_handler.return_filtered_indices(instance_ids)
        self.fov_filtered_df = self.fov_handler.fov_df.loc[indices]
        print(
            f"Number of instances {self.fov_filtered_df.index.unique(level=0).shape}, number of rows {self.fov_filtered_df.index.shape}, number of unique indexes {self.fov_filtered_df.index.unique().shape}"
        )
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

    def map_class_to_idx(self) -> Dict[str, int]:
        """One-hot encodes classes based on synsets."""
        classes = sorted(self.fov_handler.fov_df["class"].unique())
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return class_to_idx

    def map_class_to_imagenet_idx(self) -> Dict[str, int]:
        """One-hot encodes classes to their ImageNet on synsets."""
        classes = sorted(self.fov_handler.fov_df["class"].unique())
        df = self.imagenet_class_df
        class_to_idx = {
            cls_name: df[df["class_id"] == "n" + cls_name].class_idx.item()
            for cls_name in classes
        }
        return class_to_idx

    def get_sample(self, idx: int) -> Tuple[str, Tuple[float, float, float]]:
        """Selects instance_id and pose from the filtered dataframe

        Args:
            idx: index of sample to return from product(ids, fovs)

        Returns: instance_id, fov and image path
        """
        try:
            fov_vector = self.fov_filtered_df.iloc[idx].name[1:]
            # We want to order the fov as they are ordered in the fov_df
            fov_sample_dict = {
                fov_name: fov_vector[i]
                for i, fov_name in enumerate(self.fov_handler.fov_df.index.names[1:])
            }
        except IndexError as e:
            print(
                f"failed to fetch fov for {idx=}. row is {self.fov_filtered_df.iloc[idx]}"
            )
            raise e
        image_path = self.fov_filtered_df.iloc[idx]["image_path"]
        instance_id = self.fov_filtered_df.iloc[idx].name[0]
        return instance_id, fov_sample_dict, image_path

    @staticmethod
    def load_fov(data_dir: str) -> pd.DataFrame:

        tmp = {
            "Synset": str,
            "Object": str,
            "Image path": str,
        }
        types_fov = {}

        for fov in FOV_NAMES:
            if fov == "Background path":
                types_fov[fov] = str
            else:
                types_fov[fov] = float
        dtypes = {**tmp, **types_fov}
        df = pd.read_csv(
            os.path.join(data_dir, "fov_idcs.csv"), delimiter=",", dtype=dtypes
        )
        # Rename with previous naming
        df.rename(
            columns={
                "Synset": "class",
                "Object": "instance_id",
                "Image path": "image_path",
            },
            inplace=True,
        )
        df = merge_rotation_translation_single(df)
        indexes = ["instance_id"]
        for fov in FOV_NAMES:
            indexes.append(
                fov
            )  # This insured the indexes follow the names order in the fov Handler, which we use to filter indices
        # Remove unused FoV (TODO: in fact we'll always take everybody it seems)
        df = df[["class", "instance_id", "image_path"] + FOV_NAMES]
        # set columns as indices for fast filtering
        df = df.set_index(indexes)
        # sorting speeds up lookups
        df = df.sort_index()
        return df

    def get_image_info(
        self, instance_id: str, fov: Tuple[float], attribute="image_path"
    ) -> Any:
        matches = self.fov_filtered_df.loc[[(instance_id,) + fov]][
            attribute
        ].values  # DEBUG: we had a different way to do so in shapenet.py
        if len(matches) == 0:
            raise ValueError(
                f"found no matching instances for fov {fov} and instance {instance_id}"
            )
        return matches[0]

    def __getitem__(self, index: int):
        (
            instance_id,
            fov_sample_dict,
            image_path,
        ) = self.get_sample(index)
        image = self.pil_loader(image_path)

        x = self.to_tensor(image)
        label = self.get_image_info(
            instance_id, tuple(fov_sample_dict.values()), attribute="class"
        )
        label_idx = self.class_to_idx[label]
        synset = "n" + label

        fov = {
            **fov_sample_dict,
            **{
                "synset": synset,
                "class_name": self.synset_to_class_name(synset),
                "image_path": image_path,
            },
        }
        return x, label_idx, fov

    def __len__(self):
        """call index unique() to not count duplicate entries in fov csv"""
        return self.fov_filtered_df.index.unique().shape[0]


class ShapeNetExtendedDataModule(ShapeNetDataModule):
    """
    Creates training dataloaders where only a subset of instances undergo variation:
        - training dataloaders: X_canonical, X_varied
        - validation dataloaders: vary(X_canonical), held_out_canonical, held_out_varied
    """

    def __init__(
        self,
        data_dir: str,
        canonical_fov_handler: AbstractFovHandler,
        diverse_fov_handler: AbstractFovHandler,
        batch_size: int = 8,
        num_workers: int = 8,
        train_prop_to_vary: float = 0.5,
        use_imagenet_classes: bool = False,
        trainval_ids_file: str = None,
        test_ids_file: str = None,
    ):
        super(pl.LightningDataModule, self).__init__()

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

        self.canonical_fov_handler = canonical_fov_handler
        self.diverse_fov_handler = diverse_fov_handler
        # TODO I want a cleaner way to do this
        # Right now it uses the canonical fov names to rattach a fov_df to the module
        self.fov_df = ShapeNetExtended.load_fov(self.data_dir)
        self.num_classes = len(self.fov_df["class"].unique())

        if self.train_prop_to_vary > 0.0:
            self.train_loader_names = [
                "train_canonical",
                "train_diverse",
            ]
        else:
            self.train_loader_names = [
                "train_canonical",
                "train_class_not_vary_canonical",
            ]

        self.val_loader_names = ["val_canonical"] + [
            f"val_diverse_{fov}" for fov in FOV_NAMES
        ]

        self.test_loader_names = (
            ["test_canonical"]
            + [f"test_diverse_{fov}" for fov in FOV_NAMES]
            + [f"diverse_train_canonical_{fov}" for fov in FOV_NAMES]
        )

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

    def get_instance_ids_and_synsets(self) -> Tuple[List[str], List[str]]:

        """Returns a list of synsets and their corresponding instance ids."""
        synsets = (
            self.fov_df.groupby(level=0)["class"].first().values
        )  # we can't use what we had in the previous code bc 0,0,0 might not be in fov #TODO check with Mark this is correct
        instance_ids = self.fov_df.groupby(level=0)["class"].first().index
        return synsets, instance_ids

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
            print(stage)
            setattr(
                self,
                f"{stage}_canonical",
                ShapeNetExtended(
                    data_dir=self.data_dir,
                    instance_ids=canonical_ids[stage],
                    fov_handler=self.canonical_fov_handler,
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=stage_transforms,
                ),
            )

            if stage != "train":
                for fov, handler in FOV_DIVERSE_HANDLERS.items():
                    setattr(
                        self,
                        f"{stage}_diverse_{fov}",
                        ShapeNetExtended(
                            data_dir="/checkpoint/garridoq/datasets_fov/individual/",
                            instance_ids=diverse_ids[stage],
                            fov_handler=handler,
                            use_imagenet_classes=self.use_imagenet_classes,
                            img_transforms=stage_transforms,
                        ),
                    )
            else:
                # handle case where train diverse is empty
                if diverse_ids[stage]:
                    setattr(
                        self,
                        f"{stage}_diverse",
                        ShapeNetExtended(
                            data_dir=self.data_dir,
                            instance_ids=diverse_ids[stage],
                            fov_handler=self.diverse_fov_handler,
                            use_imagenet_classes=self.use_imagenet_classes,
                            img_transforms=stage_transforms,
                        ),
                    )
        for fov, handler in FOV_DIVERSE_HANDLERS.items():
            setattr(
                self,
                f"diverse_train_canonical_{fov}",
                ShapeNetExtended(
                    data_dir="/checkpoint/garridoq/datasets_fov/individual/",
                    instance_ids=train_canonical_ids,
                    fov_handler=handler,
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=self.val_augmentations,
                ),
            )

    def train_dataloader(self) -> supporters.CombinedLoader:
        if self.train_prop_to_vary > 0.0:
            loaders = {
                "train_canonical": self._make_loader(
                    self.train_canonical, num_workers=1
                ),
                "train_diverse": self._make_loader(self.train_diverse),
            }
        else:
            loaders = {
                "train_canonical": self._make_loader(
                    self.train_canonical, num_workers=1
                ),
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
                for fov in FOV_NAMES
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
                for fov in FOV_NAMES
            ]
            + [
                (
                    f"diverse_train_canonical_{fov}",
                    self._make_loader(
                        getattr(self, f"diverse_train_canonical_{fov}"), shuffle=False
                    ),
                )
                for fov in FOV_NAMES
            ]
        )
        assert (
            list(loaders.keys()) == self.test_loader_names
        ), "test loader names don't match"
        if not domain or domain == "all":
            return list(loaders.values())
        raise ValueError(f"domain {domain} not supported")


class ShapeNetClassPartitionedDataModule(ShapeNetDataModule):
    """Only some classes are diverse during training.

    Args:
        classess_to_vary: list of classes that will have varying instances.
            Classes not in this list will only contain canonical samples.
        train_prop_to_vary: for classes the vary, this determines the proportion of instances within the class that will vary.
    """

    def __init__(
        self,
        data_dir: str,
        canonical_fov_handler: AbstractFovHandler,
        diverse_fov_handler: AbstractFovHandler,
        batch_size: int = 8,
        num_workers: int = 8,
        train_prop_to_vary: float = 0.5,
        use_imagenet_classes: bool = False,
        trainval_ids_file: str = None,
        test_ids_file: str = None,
        classes_to_vary: List[str] = ["03691459", "04074963"],
    ):

        super(pl.LightningDataModule, self).__init__()

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

        self.canonical_fov_handler = canonical_fov_handler
        self.diverse_fov_handler = diverse_fov_handler
        self.fov_df = ShapeNetExtended.load_fov(self.data_dir)
        self.num_classes = len(self.fov_df["class"].unique())
        self.classes_to_vary = classes_to_vary

        if self.train_prop_to_vary > 0.0:
            self.train_loader_names = [
                "train_class_vary_canonical",
                "train_class_not_vary_canonical",
                "train_class_vary_diverse",
            ]
        else:
            self.train_loader_names = [
                "test_class_vary_canonical",
            ]

        self.val_loader_names = ["val_canonical"] + [
            f"val_diverse_{fov}" for fov in FOV_NAMES
        ]

        self.val_loader_names += ["val_class_vary_canonical"] + [
            f"val_class_vary_diverse_{fov}" for fov in FOV_NAMES
        ]

        self.val_loader_names += ["val_class_not_vary_canonical"] + [
            f"val_class_not_vary_diverse_{fov}" for fov in FOV_NAMES
        ]

        self.test_loader_names = (
            ["test_canonical"]
            + [f"test_diverse_{fov}" for fov in FOV_NAMES]
            + [f"diverse_train_class_vary_canonical_{fov}" for fov in FOV_NAMES]
            + [f"diverse_train_class_not_vary_{fov}" for fov in FOV_NAMES]
        )

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

    def get_instance_ids_and_synsets(self) -> Tuple[List[str], List[str]]:

        """Returns a list of synsets and their corresponding instance ids."""
        synsets = (
            self.fov_df.groupby(level=0)["class"].first().values
        )  # we can't use what we had in the previous code bc 0,0,0 might not be in fov #TODO check with Mark this is correct
        instance_ids = self.fov_df.groupby(level=0)["class"].first().index
        return synsets, instance_ids

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

        print(f"{len(train_canonical_ids)=} {len(train_varied_ids)=}")
        (
            train_class_vary_canonical_ids,
            train_class_not_vary_canonical_ids,
        ) = self.filter_class_vary(
            synset_to_instance_ids, set(train_canonical_ids), self.classes_to_vary
        )
        (
            train_class_vary_varied_ids,
            train_class_not_vary_varied_ids,
        ) = self.filter_class_vary(
            synset_to_instance_ids, set(train_varied_ids), self.classes_to_vary
        )

        print(
            f"{len(train_class_vary_canonical_ids)=} {len(train_class_not_vary_canonical_ids)=}"
        )

        # Concatenate all canonical and diverse into 1 list for classes that do not vary
        train_class_not_vary_ids = (
            train_class_not_vary_canonical_ids + train_class_not_vary_varied_ids
        )

        val_class_vary_ids, val_class_not_vary_ids = self.filter_class_vary(
            synset_to_instance_ids, set(val_ids), self.classes_to_vary
        )

        assert (
            len(
                np.intersect1d(train_class_vary_canonical_ids, train_class_not_vary_ids)
            )
            == 0
        )
        assert (
            len(np.intersect1d(train_class_vary_varied_ids, train_class_not_vary_ids))
            == 0
        )
        self.create_datasets(
            train_class_vary_canonical_ids,
            train_class_vary_varied_ids,
            train_class_not_vary_ids,
            val_ids,
            val_class_vary_ids,
            val_class_not_vary_ids,
            test_ids,
        )

    def create_datasets(
        self,
        train_class_vary_canonical_ids: List[str],
        train_class_vary_varied_ids: List[str],
        train_class_not_vary_ids: List[str],
        val_ids: List[str],
        val_class_vary_ids: List[str],
        val_class_not_vary_ids: List[str],
        test_ids: List[str],
    ):
        """Creates datasets for eval"""
        canonical_ids = {
            "train_class_vary": train_class_vary_canonical_ids,
            "train_class_not_vary": train_class_not_vary_ids,
            "val": val_ids,
            "val_class_vary": val_class_vary_ids,
            "val_class_not_vary": val_class_not_vary_ids,
            "test": test_ids,
        }
        diverse_ids = {
            "train_class_vary": train_class_vary_varied_ids,
            "train_class_not_vary": None,  # we have no TRAINING diverse ids for classes that do not vary
            "val": val_ids,
            "val_class_vary": val_class_vary_ids,
            "val_class_not_vary": val_class_not_vary_ids,
            "test": test_ids,
        }

        for stage in [
            "train_class_vary",
            "train_class_not_vary",
            "val",
            "val_class_vary",
            "val_class_not_vary",
            "test",
        ]:
            stage_transforms = (
                self.train_augmentations
                if stage in ["train_class_vary", "train_class_not_vary"]
                else self.val_augmentations
            )
            print(stage)
            setattr(
                self,
                f"{stage}_canonical",
                ShapeNetExtended(
                    data_dir=self.data_dir,
                    instance_ids=canonical_ids[stage],
                    fov_handler=self.canonical_fov_handler,
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=stage_transforms,
                ),
            )

            if stage not in ["train_class_vary", "train_class_not_vary"]:
                for fov, handler in FOV_DIVERSE_HANDLERS.items():
                    setattr(
                        self,
                        f"{stage}_diverse_{fov}",
                        ShapeNetExtended(
                            data_dir="/checkpoint/garridoq/datasets_fov/individual/",
                            instance_ids=diverse_ids[stage],
                            fov_handler=handler,
                            use_imagenet_classes=self.use_imagenet_classes,
                            img_transforms=stage_transforms,
                        ),
                    )
            else:
                # handle case where train diverse is empty
                if diverse_ids[stage]:
                    setattr(
                        self,
                        f"{stage}_diverse",
                        ShapeNetExtended(
                            data_dir=self.data_dir,
                            instance_ids=diverse_ids[stage],
                            fov_handler=self.diverse_fov_handler,
                            use_imagenet_classes=self.use_imagenet_classes,
                            img_transforms=stage_transforms,
                        ),
                    )
        for fov, handler in FOV_DIVERSE_HANDLERS.items():
            setattr(
                self,
                f"diverse_train_class_vary_canonical_{fov}",
                ShapeNetExtended(
                    data_dir="/checkpoint/garridoq/datasets_fov/individual/",
                    instance_ids=train_class_vary_canonical_ids,
                    fov_handler=handler,
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=self.val_augmentations,
                ),
            )
            setattr(
                self,
                f"diverse_train_class_not_vary_{fov}",
                ShapeNetExtended(
                    data_dir="/checkpoint/garridoq/datasets_fov/individual/",
                    instance_ids=train_class_not_vary_ids,
                    fov_handler=handler,
                    use_imagenet_classes=self.use_imagenet_classes,
                    img_transforms=self.val_augmentations,
                ),
            )

    def filter_class_vary(
        self,
        class_to_instances: Dict[str, List[str]],
        ids_to_filter: Set[str],
        class_to_vary: List[str],
    ):
        """Partitions instance ids accross varying / non varying classes

        Args:
            class_to_instances: synset str -> [instance_ids]
            ids_to_filter: set of instance ids to filter and add to either vary/not vary
        """
        class_vary_ids = []
        class_not_vary_ids = []
        for synset, instance_ids in class_to_instances.items():
            if synset in class_to_vary:
                class_vary_ids += [
                    i for i in ids_to_filter if i in instance_ids
                ]  # Add the ones in the set to filter to the class_vary ones
            else:
                class_not_vary_ids += [i for i in ids_to_filter if i in instance_ids]
        return class_vary_ids, class_not_vary_ids

    def train_dataloader(self) -> supporters.CombinedLoader:
        """Returns one loader with the diverse instances and another with canonical instances"""
        if self.train_prop_to_vary > 0.0:
            loaders = {
                "train_class_vary_canonical": self._make_loader(
                    self.train_class_vary_canonical, num_workers=1
                ),
                "train_class_vary_diverse": self._make_loader(
                    self.train_class_vary_diverse
                ),
                "train_class_not_vary_canonical": self._make_loader(
                    self.train_class_not_vary_canonical, num_workers=1
                ),
            }
        else:
            loaders = {
                "train_class_vary_canonical": self._make_loader(
                    self.train_class_vary_canonical, num_workers=1
                ),
                "train_class_not_vary_canonical": self._make_loader(
                    self.train_class_not_vary_canonical, num_workers=1
                ),
            }
        # max_size_cycle cycles through canonical poses to match size of diverse poses
        combined_loaders = supporters.CombinedLoader(loaders, mode="max_size_cycle")
        return combined_loaders

    def val_dataloader(self) -> List[DataLoader]:
        """
        val_loaders should separate canonical versus diverse classes.

        Returns:
        - val_canonical: dataloader with all instances in their canonical pose
        - val_canonical_for_diverse_classes: dataloader with instances from classes_to_vary in their canonical view
        - val_canonical_for_canonical_classes: dataloader with instances
            from classes NOT in classes_to_vary in their canonical pose

        For each individual factor:
        - val_diverse_[Factor]: dataloader with all instances in diverse [Factor] values.
        - val_diverse_[Factor]_for_diverse_classes: dataloader with instances from classes_to_vary in diverse [Factor] values
        - val_diverse_[Factor]_for_canonical_classes: dataloader with instances NOT from classes_to_vary in diverse [Factor] values
        """

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
                for fov in FOV_NAMES
            ]
            + [
                (
                    "val_class_vary_canonical",
                    self._make_loader(
                        self.val_class_vary_canonical, shuffle=False, num_workers=1
                    ),
                )
            ]
            + [
                (
                    f"val_class_vary_diverse_{fov}",
                    self._make_loader(
                        getattr(self, f"val_class_vary_diverse_{fov}"),
                        shuffle=False,
                    ),
                )
                for fov in FOV_NAMES
            ]
            + [
                (
                    "val_class_not_vary_canonical",
                    self._make_loader(
                        self.val_class_not_vary_canonical, shuffle=False, num_workers=1
                    ),
                )
            ]
            + [
                (
                    f"val_class_not_vary_diverse_{fov}",
                    self._make_loader(
                        getattr(self, f"val_class_not_vary_diverse_{fov}"),
                        shuffle=False,
                    ),
                )
                for fov in FOV_NAMES
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
                for fov in FOV_NAMES
            ]
            + [
                (
                    f"diverse_train_class_vary_canonical_{fov}",
                    self._make_loader(
                        getattr(self, f"diverse_train_class_vary_canonical_{fov}"),
                        shuffle=False,
                    ),
                )
                for fov in FOV_NAMES
            ]
            + [
                (
                    f"diverse_train_class_not_vary_{fov}",
                    self._make_loader(
                        getattr(self, f"diverse_train_class_not_vary_{fov}"),
                        shuffle=False,
                    ),
                )
                for fov in FOV_NAMES
            ]
        )
        assert (
            list(loaders.keys()) == self.test_loader_names
        ), "test loader names don't match"
        if not domain or domain == "all":
            return list(loaders.values())
        raise ValueError(f"domain {domain} not supported")


class ShapeNetIndividualFactorDataModule(ShapeNetExtendedDataModule):
    """A single factor varies.
    If subset is true, then only 20 instances are loaded for train/val and test
    """

    def __init__(
        self,
        train_prop_to_vary: float = 0.5,
        factor_to_vary: str = "Rotation",
        batch_size: int = 8,
        num_workers: int = 4,
        data_dir="/checkpoint/garridoq/datasets_fov/individual",
        subset: bool = False,
    ):
        self.factor_to_vary = factor_to_vary
        assert factor_to_vary in FOV_DIVERSE_HANDLERS, f"{factor_to_vary} not supported"
        super().__init__(
            data_dir=data_dir,
            canonical_fov_handler=SingleFovHandler({}),
            diverse_fov_handler=FOV_DIVERSE_HANDLERS[factor_to_vary],
            trainval_ids_file=TRAIN_VAL_IDS_SMALL_PATH
            if subset
            else TRAIN_VAL_IDS_PATH,
            test_ids_file=TEST_IDS_SMALL_PATH if subset else TEST_IDS_PATH,
            train_prop_to_vary=train_prop_to_vary,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class ShapeNetPairedFactorDataModule(ShapeNetExtendedDataModule):
    """A pair of factors vary.
    If subset is true, then only 20 instances are loaded for train/val and test
    """

    FOV_TO_VALUES = {
        "Rotation": np.arange(0, 101, 10),
        "Translation": np.arange(0, 101, 10),
        "Scale": np.arange(0, 101, 10),
        "Spot hue": np.arange(0, 101, 10),
        "Background": BACKGROUND_PATHS,
    }

    def __init__(
        self,
        train_prop_to_vary: float = 0.5,
        factors_to_vary: Tuple[str, str] = ("Rotation", "Translation"),
        batch_size: int = 8,
        num_workers: int = 4,
        data_dir="/checkpoint/garridoq/datasets_fov/pairs",
        subset: bool = False,
    ):
        self.factors_to_vary = factors_to_vary

        fov1, fov2 = self.factors_to_vary
        fov_to_values = {fov1: self.FOV_TO_VALUES[fov1], fov2: self.FOV_TO_VALUES[fov2]}
        diverse_fov_handler = PairFovHandler(fov_to_values, fov_1=fov1, fov_2=fov2)

        super().__init__(
            data_dir=data_dir,
            canonical_fov_handler=SingleFovHandler({}),
            diverse_fov_handler=diverse_fov_handler,
            trainval_ids_file=TRAIN_VAL_IDS_SMALL_PATH
            if subset
            else TRAIN_VAL_IDS_PATH,
            test_ids_file=TEST_IDS_SMALL_PATH if subset else TEST_IDS_PATH,
            train_prop_to_vary=train_prop_to_vary,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class ShapeNetAllFactorsDataModule(ShapeNetExtendedDataModule):
    """All factors vary.
    If subset is true, then only 20 instances are loaded for train/val and test
    """

    def __init__(
        self,
        train_prop_to_vary: float = 0.5,
        batch_size: int = 8,
        num_workers: int = 4,
        data_dir="/checkpoint/garridoq/datasets_fov/random",
        subset: bool = False,
    ):
        diverse_fov_handler = AllFovHandler(
            {},
            fov_indices_file="/checkpoint/garridoq/datasets_fov/random/fov_combinations_idcs.npy",
        )

        super().__init__(
            data_dir=data_dir,
            canonical_fov_handler=SingleFovHandler({}),
            diverse_fov_handler=diverse_fov_handler,
            trainval_ids_file=TRAIN_VAL_IDS_SMALL_PATH
            if subset
            else TRAIN_VAL_IDS_PATH,
            test_ids_file=TEST_IDS_SMALL_PATH if subset else TEST_IDS_PATH,
            train_prop_to_vary=train_prop_to_vary,
            batch_size=batch_size,
            num_workers=num_workers,
        )


class ShapeNetRandomClassSplitDataModule(ShapeNetClassPartitionedDataModule):
    """Wrapper for class partition to generate random splits"""

    # subset of 27 classes to vary
    synset_to_name_subset = {
        "02876657": "bottle",
        "03928116": "piano;pianoforte;forte-piano",
        "03710193": "mailbox;letter box",
        "03593526": "jar",
        "03938244": "pillow",
        "04460130": "tower",
        "04554684": "washer;automatic washer;washing machine",
        "04401088": "telephone;phone;telephone set",
        "02773838": "bag;traveling bag;travelling bag;grip;suitcase",
        "03211117": "display;video display",
        "02880940": "bowl",
        "03759954": "microphone;mike",
        "04256520": "sofa;couch;lounge",
        "03790512": "motorcycle;bike",
        "03513137": "helmet",
        "02818832": "bed",
        "04530566": "vessel;watercraft",
        "03761084": "microwave;microwave oven",
        "02843684": "birdhouse",
        "02946921": "can;tin;tin can",
        "03001627": "chair",
        "02958343": "car;auto;automobile;machine;motorcar",
        "02691156": "airplane;aeroplane;plane",
        "04379243": "table",
        "04074963": "remote control;remote",
        "03948459": "pistol;handgun;side arm;shooting iron",
        "03325088": "faucet;spigot",
    }

    def __init__(
        self,
        train_prop_to_vary: float = 0.5,
        batch_size: int = 8,
        num_workers: int = 4,
        data_dir="/checkpoint/garridoq/datasets_fov/individual",
        factor_to_vary: str = "Rotation",
    ):

        classes_to_vary = list(self.synset_to_name_subset.keys())

        test_ids_path = (
            "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/test_ids.txt"
        )
        train_val_ids_path = (
            "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/trainval_ids.txt"
        )

        canonical_fovs = SingleFovHandler({})
        super().__init__(
            data_dir=data_dir,
            canonical_fov_handler=canonical_fovs,
            diverse_fov_handler=FOV_DIVERSE_HANDLERS[factor_to_vary],
            trainval_ids_file=train_val_ids_path,
            test_ids_file=test_ids_path,
            train_prop_to_vary=train_prop_to_vary,
            batch_size=batch_size,
            num_workers=num_workers,
            classes_to_vary=classes_to_vary,
        )
