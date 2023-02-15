from datasets.dummy import DummyDataModule

from datasets.image_datamodule import ImageDataModule
from datasets.imagenet_rendition import ImageNetRenditionDataModule
from datasets.imagenet_adversarial import ImageNetAdversarialDataModule
from datasets.objectnet import ObjectNetDataModule
from datasets.imagenet_sketch import ImageNetSketchDataModule
from datasets.imagenet_1k import ImageNet1kDataModule

from datasets.dollarstreet import DollarStreetDataModule
from pathlib import Path
import pytest


class TestImageNet:
    def test_imagenet(self):
        dm = ImageNet1kDataModule(batch_size=8)
        assert Path(dm.data_dir, "val").exists()
        test_batch = dm.test_dataloader()
        x, y = next(iter(test_batch))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)


class TestImageNetV2:
    def test_imagenetv2(self):
        dm = ImageDataModule(
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


class TestImageNetA:
    def test_imageneta(self):
        dm = ImageNetAdversarialDataModule(
            batch_size=8,
        )
        assert Path(dm.data_dir, "test").exists()
        test_batch = dm.test_dataloader()
        x, y = next(iter(test_batch))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)


class TestImageNetSketch:
    def test_imagenetsketch(self):
        dm = ImageNetSketchDataModule(
            batch_size=8,
        )
        assert Path(dm.data_dir, "test").exists()
        test_batch = dm.test_dataloader()
        x, y = next(iter(test_batch))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)


class TestObjectNet:
    def test_objectnet(self):
        dm = ObjectNetDataModule(
            batch_size=8,
        )
        assert Path(dm.data_dir, "test").exists()
        test_batch = dm.test_dataloader()
        x, y = next(iter(test_batch))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)


class TestDollarStreet:
    def test_dollarstreet(self):
        dm = DollarStreetDataModule(batch_size=8)
        assert Path(dm.data_dir, "test").exists()
        test_loader = dm.test_dataloader()
        x, y, url = next(iter(test_loader))

        assert x.shape == (8, 3, 224, 224)
        assert len(y) == 8


@pytest.mark.webtest
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
