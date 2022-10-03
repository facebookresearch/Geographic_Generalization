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
from datasets.shapenet import shapenet_extended, shapenet_v2
import os


class TestShapeNetExtended:
    DATA_DIR = "/checkpoint/garridoq/datasets_fov/"
    TEST_IDS_PATH = (
        "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/test_ids.txt"
    )
    TRAIN_VAL_IDS_PATH = (
        "/checkpoint/marksibrahim/datasets/multi_factor_shapenet/trainval_ids.txt"
    )
    SETTINGS = ["individual", "pairs", "random"]

    @pytest.mark.slow
    def test_individual_factor_datamodule(self):
        data_dir = os.path.join(self.DATA_DIR, "individual")
        canonical_fovs = SingleFovHandler({})
        factor = "Translation"
        dm = shapenet_extended.ShapeNetExtendedDataModule(
            data_dir=data_dir,
            canonical_fov_handler=canonical_fovs,
            diverse_fov_handler=FOV_DIVERSE_HANDLERS[factor],
            trainval_ids_file=self.TRAIN_VAL_IDS_PATH,
            test_ids_file=self.TEST_IDS_PATH,
            train_prop_to_vary=0.5,
        )
        dm.setup()
        train_loaders = dm.train_dataloader()
        assert type(train_loaders) is supporters.CombinedLoader
        val_loaders = dm.val_dataloader()
        assert len(val_loaders) > 0
        test_loaders = dm.test_dataloader()
        assert len(test_loaders) > 0

    @pytest.mark.slow
    def test_individual_factor_w_zero_train_diversity_datamodule(self):
        data_dir = os.path.join(self.DATA_DIR, "individual")
        canonical_fovs = SingleFovHandler({})
        factor = "Translation"
        dm = shapenet_extended.ShapeNetExtendedDataModule(
            data_dir=data_dir,
            canonical_fov_handler=canonical_fovs,
            diverse_fov_handler=FOV_DIVERSE_HANDLERS[factor],
            trainval_ids_file=self.TRAIN_VAL_IDS_PATH,
            test_ids_file=self.TEST_IDS_PATH,
            train_prop_to_vary=0.0,
        )
        dm.setup()
        train_loaders = dm.train_dataloader()
        assert type(train_loaders) is supporters.CombinedLoader
        val_loaders = dm.val_dataloader()
        assert len(val_loaders) > 0
        test_loaders = dm.test_dataloader()
        assert len(test_loaders) > 0

    @pytest.mark.slow
    def test_ShapeNetIndividualFactorDataModule(self):
        dm = shapenet_extended.ShapeNetIndividualFactorDataModule(
            train_prop_to_vary=0.5, factor_to_vary="Rotation"
        )
        dm = shapenet_extended.ShapeNetIndividualFactorDataModule(
            train_prop_to_vary=0.5, factor_to_vary="Background path"
        )
        dm.setup()
        # test setup of the 0.0 diversity corner case
        dm = shapenet_extended.ShapeNetIndividualFactorDataModule(
            train_prop_to_vary=0.0, factor_to_vary="Rotation"
        )
        dm.setup()

    @pytest.mark.slow
    def test_ShapeNetPairedFactorDataModule(self):
        dm = shapenet_extended.ShapeNetPairedFactorDataModule(
            train_prop_to_vary=0.5, factors_to_vary=("Rotation", "Translation")
        )
        dm.setup()

    @pytest.mark.slow
    def test_ShapeNetAllFactorsDataModule(self):
        dm = shapenet_extended.ShapeNetAllFactorsDataModule(
            train_prop_to_vary=0.5, subset=True
        )
        dm.setup()

    @pytest.mark.slow
    def test_ShapeNetRandomClassSplitDataModule_translation(self):
        factor = "Translation"
        dm = shapenet_extended.ShapeNetRandomClassSplitDataModule(factor_to_vary=factor)
        dm.setup()
        assert len(dm.classes_to_vary) > 0

        train_loaders = dm.train_dataloader()
        assert type(train_loaders) is pl.CombinedLoader

    @pytest.mark.slow
    def test_ShapeNetRandomClassSplitDataModule_rotation(self):
        factor = "Rotation"
        dm = shapenet_extended.ShapeNetRandomClassSplitDataModule(factor_to_vary=factor)
        dm.setup()
        assert len(dm.classes_to_vary) > 0

        train_loaders = dm.train_dataloader()
        assert type(train_loaders) is pl.CombinedLoader


class TestShapeNetv2Datasets:
    instance_ids = [
        "10155655850468db78d106ce0a280f87",
        "1021a0914a7207aff927ed529ad90a11",
    ]

    ds = shapenet_v2.ShapeNetSampleVariationPerInstance(
        instance_ids=instance_ids, factor_to_vary="Rotation", train_prop_to_vary=0.5
    )

    def test_num_rows_total(self):
        num_instances = len(self.instance_ids)
        # canonical + instances * 4 factors * 100 parameters + instances * background (24)
        expected_num_rows = (
            num_instances + (num_instances * 4 * 100) + (num_instances * 24)
        )
        assert len(self.ds.df) == expected_num_rows

    def test_num_rows_for_factors(self):
        num_instances = len(self.instance_ids)
        assert self.ds.df["is_canonical"].sum() == num_instances
        # 101 parameters including canonical
        assert len(self.ds.df["Rotation X"].unique()) == 101
        assert len(self.ds.df["Translation Y"].unique()) == 101
        assert len(self.ds.df["Scale"].unique()) == 101
        assert len(self.ds.df["Spot hue"].unique()) == 101
        assert len(self.ds.df["Background path"].unique()) == 25

    def test_factor_varying(self):
        assert len(self.ds.factor_df["Rotation Y"].unique()) == 101
        assert len(self.ds.factor_df["Translation Y"].unique()) == 1
        assert len(self.ds.factor_df["Background path"].unique()) == 1

    def test_get_item(self):
        x, y, fov = self.ds[1]
        assert x.shape == (3, 224, 224)
        assert type(y) is int
        assert "image_path" in fov
        assert fov["instance_id"] == self.instance_ids[1]

    def test_all_diverse(self):
        num_instances = len(self.instance_ids)
        ds = shapenet_v2.ShapeNetAllDiverse(
            instance_ids=self.instance_ids, factor_to_vary="Rotation"
        )
        assert len(ds.factor_df) == 101 * num_instances
        x, y, fov = ds[1]
        assert x.shape == (3, 224, 224)
        assert type(y) is int
        assert "image_path" in fov

    def test_all_canonical(self):
        num_instances = len(self.instance_ids)
        ds = shapenet_v2.ShapeNetAllCanonical(
            instance_ids=self.instance_ids, factor_to_vary="Rotation"
        )
        assert len(ds.factor_df) == 101 * num_instances
        x, y, fov = ds[1]
        assert x.shape == (3, 224, 224)
        assert type(y) is int
        assert "image_path" in fov
        assert fov["is_canonical"] == True


class TestShapeNetv2DataModule:
    train_prop_to_vary = 0.5
    batch_size = 16
    dm = shapenet_v2.ShapeNetv2SingleFactorDataModule(
        train_prop_to_vary=train_prop_to_vary, batch_size=batch_size
    )

    def test_id_splits(self):
        assert len(self.dm.train_ids) == 1890
        assert len(self.dm.val_ids) == 405
        assert len(self.dm.test_ids) == 405

    def test_datasets(self):
        self.dm.setup()
        assert len(self.dm.all_fov_df) > 1000000

        assert len(self.dm.train_dataset) == 1890
        assert len(self.dm.val_canonical) == 405
        assert len(self.dm.test_canonical) == 405

        assert len(self.dm.val_diverse_Rotation) == 101 * 405
        assert len(self.dm.val_diverse_Translation) == 101 * 405
        assert len(getattr(self.dm, "val_diverse_Background path")) == 25 * 405

    def test_dataloaders(self):
        self.dm.setup()

        train_loaders = self.dm.train_dataloader()
        expected_len = -(1890 // -self.batch_size)
        assert len(train_loaders) == expected_len

        val_loaders = self.dm.val_dataloader()
        expected_len = -(405 // -self.batch_size)
        assert len(val_loaders[0]) == expected_len
        expected_len = -((405 * 101) // -self.batch_size)
        assert len(val_loaders[2]) == expected_len
