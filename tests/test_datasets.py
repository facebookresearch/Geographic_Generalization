from datasets.dummy import DummyDataModule
from datasets.imagenet import ImageNetDataModule


class TestImageNet:
    def test_imagenet(self):
        dm = ImageNetDataModule(batch_size=8)
        val_batch = dm.val_dataloader()
        x, y = next(iter(val_batch))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)


class TestDummy:
    def test_dummy(self):
        dm = DummyDataModule(batch_size=8)
        val_loader = dm.val_dataloader()
        train_loader = dm.train_dataloader()
        self.confirm_sizes(val_loader)
        self.confirm_sizes(train_loader)

    def confirm_sizes(self, loader):
        x, y = next(iter(loader))
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8,)
