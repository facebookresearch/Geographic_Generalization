from torch.utils.data import DataLoader, Dataset
import pytest
import pytorch_lightning as pl
from pytorch_lightning.trainer import supporters
import torch

from datasets.shapenet.shapenet_extended import (
    SingleFovHandler,
    PairFovHandler,
    AllFovHandler,
    FOV_DIVERSE_HANDLERS,
)
from datasets.shapenet import shapenet_extended
import os

DATA_DIR = "/checkpoint/garridoq/datasets_fov/"
TEST_IDS_PATH = (
    "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/test_ids.txt"
)
TRAIN_VAL_IDS_PATH = (
    "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/trainval_ids.txt"
)
SETTINGS = ["individual", "pairs", "random"]

data_dir = os.path.join(DATA_DIR, "individual")
canonical_fovs = SingleFovHandler({})
factor = "Rotation"
dm = shapenet_extended.ShapeNetClassPartitionedDataModule(
        data_dir=data_dir,
        canonical_fov_handler=canonical_fovs,
        diverse_fov_handler=FOV_DIVERSE_HANDLERS[factor],
        trainval_ids_file=TRAIN_VAL_IDS_PATH,
        test_ids_file=TEST_IDS_PATH,
        train_prop_to_vary=0.5
    )
dm.setup()
train_loaders = dm.train_dataloader()
assert type(train_loaders) is supporters.CombinedLoader
val_loaders = dm.val_dataloader()
assert len(val_loaders) > 0
test_loaders = dm.test_dataloader()
assert len(test_loaders) > 0