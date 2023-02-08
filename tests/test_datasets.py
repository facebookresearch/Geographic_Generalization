from datasets.dummy import DummyDataModule
from datasets.imagenet import ImageNetDataModule
from datasets.dollarstreet import DollarStreetDataModule
from pathlib import Path


class TestImageNet:
    def test_imagenet(self):
        dm = ImageNetDataModule(batch_size=8)
        assert Path(dm.data_dir, "test").exists()
        test_loader = dm.test_dataloader()
        x, y = next(iter(test_loader))
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
