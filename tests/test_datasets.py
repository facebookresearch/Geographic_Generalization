from datasets.dummy import DummyDataModule
from datasets.imagenet import ImageNetDataModule
from datasets.imagenet_rendition import ImageNetRenditionDataModule
from pathlib import Path


class TestImageNet:
    def test_imagenet(self):
        dm = ImageNetDataModule(batch_size=8)
        assert Path(dm.data_dir, "test").exists()
        test_batch = dm.test_dataloader()
        x, y = next(iter(test_batch))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)


class TestImageNetV2:
    def test_imagenetv2(self):
        dm = ImageNetDataModule(
            data_dir="/checkpoint/meganrichards/datasets/imagenetv2-matched-frequency/",
            batch_size=8,
        )
        assert Path(dm.data_dir, "test").exists()
        test_batch = dm.test_dataloader()
        x, y = next(iter(test_batch))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)


class TestImageNetR:
    def test_imagenetv2(self):
        dm = ImageNetRenditionDataModule(
            batch_size=8,
        )
        assert Path(dm.data_dir, "test").exists()
        test_batch = dm.test_dataloader()
        x, y = next(iter(test_batch))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)


class TestDummy:
    def test_dummy(self):
        dm = DummyDataModule(batch_size=8, num_samples=50)
        val_loader = dm.val_dataloader()
        train_loader = dm.train_dataloader()
        self.confirm_sizes(val_loader)
        self.confirm_sizes(train_loader)

    def confirm_sizes(self, loader):
        x, y = next(iter(loader))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)
