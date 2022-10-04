from datasets.imagenet import ImageNetDataModule


class TestImageNet:
    def test_imagenet(self):
        dm = ImageNetDataModule(batch_size=8)
        val_batch = dm.val_dataloader()[0]
        x, y = val_batch
        assert x.shape == (8, 3, 224, 224)
        assert y.shape == (8, 1)
