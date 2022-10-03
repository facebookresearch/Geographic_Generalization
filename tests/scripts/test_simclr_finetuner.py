import pytorch_lightning as pl
from pl_bolts.models.self_supervised import SimCLR
from pytorch_lightning.loggers.wandb import WandbLogger
from models.finetuning.linear_finetuner import LinearFineTuner
from datasets.imagenet import ImageNetSimCLRDataModule
import submitit
import click
import os


def fine_tune(gpus: int = 8, local: bool = True):
    weight_path = "https://pl-bolts-weights.s3.us-east-2.amazonaws.com/simclr/bolts_simclr_imagenet/simclr_imagenet.ckpt"
    simclr = SimCLR.load_from_checkpoint(weight_path, strict=False)

    simclr.freeze()
    simclr.embedding_dim = 2048
    model = LinearFineTuner(simclr, num_classes=1000, learning_rate=0.6)
    dm = ImageNetSimCLRDataModule(batch_size=64)
    logger = WandbLogger(
        entity="video-variation",
        project="tmp",
        name="simclr-pretrained",
    )
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=100,
        logger=logger,
        strategy="ddp" if gpus > 1 else None,
        # limit_train_batches=10,
        # limit_val_batches=2,
        # log_every_n_steps=1,
    )
    trainer.fit(model, dm)
    print(f"{model.train_accuracy.compute().item()=}")

    print(f"{model.val_accuracy.compute().item()=}")


@click.command()
@click.option("--gpus", default=8)
@click.option("--local", default=True, help="Run on cluster or locally")
@click.option("--partition", default="dev", help="Cluster partition: dev")
def run(gpus, local, partition):
    if local:
        fine_tune(gpus=gpus, local=local)
    else:
        user = os.environ["USER"]
        executor = submitit.SlurmExecutor(
            folder=f"/checkpoint/{user}/logs/tmp/simclr_finetuner_test"
        )
        executor.update_parameters(
            time=3000,
            gpus_per_node=gpus,
            ntasks_per_node=gpus,
            cpus_per_task=8,
            partition=partition,
            # default unit is MB
            mem=490 * 1000,
        )
        job = executor.submit(fine_tune, gpus, local)
        print(job.result())


if __name__ == "__main__":
    run()
