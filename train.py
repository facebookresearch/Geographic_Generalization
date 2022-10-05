import hydra
from submitit.helpers import RsyncSnapshot
from hydra.utils import instantiate
from omegaconf import DictConfig
from models.loggers import (
    setup_wandb,
    print_config,
    get_git_hash,
    find_existing_checkpoint,
)
import os
import pytorch_lightning as pl
import tempfile
from pytorch_lightning.plugins.environments import SLURMEnvironment
import logging

log = logging.getLogger(__name__)
git_hash = get_git_hash()


@hydra.main(
    version_base="1.2", config_path="config", config_name="train_defaults.yaml"
)
def main(config: DictConfig) -> None:
    print_config(config)
    pl.seed_everything(config.seed)
    wandb_logger = setup_wandb(config, log, git_hash)
    job_logs_dir = os.getcwd()

    datamodule = instantiate(config.datamodule)

    model = instantiate(config.module)

    trainer = pl.Trainer(
        **config.trainer,
        plugins=SLURMEnvironment(auto_requeue=False),
        logger=wandb_logger,
    )

    resume_ckpt = find_existing_checkpoint(job_logs_dir)
    trainer.fit(model, datamodule=datamodule, ckpt_path=resume_ckpt)
    trainer.validate(datamodule=datamodule)
    # trainer.test(datamodule=datamodule)

    # allows for logging separate experiments with multi-run (-m) flag
    wandb_logger.experiment.finish()


if __name__ == "__main__":
    user = os.getlogin()
    print("git hash: ", git_hash)
    snapshot_dir = tempfile.mkdtemp(prefix=f"/checkpoint/{user}/tmp/")
    print("Snapshot dir is: ", snapshot_dir)
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()
