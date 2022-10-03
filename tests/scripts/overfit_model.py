import pytorch_lightning as pl
from datasets.shapenet import ShapeNetPairsDataModule
from models.mae.mae_shapenet import MAELieModule, MAELieMaskedModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import DeviceStatsMonitor


def overfit(model, datamodule):
    logger = WandbLogger(project="tmp", name="overfit-test")
    trainer = pl.Trainer(
        gpus="1",
        overfit_batches=10,
        max_epochs=5,
        log_every_n_steps=5,
        logger=logger,
        # callbacks=[DeviceStatsMonitor()],
    )
    trainer.fit(model, datamodule=datamodule)


def shapenet_pairs_small():
    datamodule = ShapeNetPairsDataModule(
        data_dir="/checkpoint/marksibrahim/datasets/shapenet_renderings_overlapping_small",
        use_imagenet_classes=False,
        batch_size=2,
        num_workers=1,
    )
    return datamodule


if __name__ == "__main__":
    datamodule = shapenet_pairs_small()
    model = MAELieModule(datamodule=datamodule)
    # model = MAELieMaskedModule(datamodule=datamodule)
    overfit(model, datamodule)
