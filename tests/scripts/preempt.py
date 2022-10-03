"""
Runs experiments on local or slurm cluster

python train_video_classifier.py -m 
python train_video_classifier.py -m mode=local

To run a specific experiment:
python train_video_classifier.py -m +experiment=kinetics400_resnet_3d_classifier
"""

import hydra
import tempfile
import logging
import pytorch_lightning as pl
import os
import submitit

from pytorch_lightning.plugins.environments import SLURMEnvironment
from hydra.utils import instantiate
from submitit.helpers import RsyncSnapshot
from omegaconf import DictConfig
from models.loggers import (
    setup_best_val_checkpoint,
    setup_last_val_checkpoint,
    setup_last_checkpoint,
    VideoClassificationLogger,
    setup_wandb,
)

log = logging.getLogger(__name__)


@hydra.main(config_path="../config", config_name="video_classifier_defaults.yaml")
def main(config: DictConfig):
    env = submitit.JobEnvironment()
    log.info(f"{env}")

    pl.seed_everything(config.seed)
    data_module = instantiate(config.data_module)
    wandb_logger = setup_wandb(config, log)
    job_logs_dir = os.getcwd()

    model = instantiate(config.module)

    trainer = pl.Trainer(
        **config.trainer,
        plugins=SLURMEnvironment(auto_requeue=False),
        logger=wandb_logger,
        callbacks=[
            VideoClassificationLogger(data_module),
            setup_last_checkpoint(dirpath=job_logs_dir, model_name=model.model_name),
            setup_last_val_checkpoint(dirpath=job_logs_dir),
            setup_best_val_checkpoint(
                model_name=model.model_name, dirpath=job_logs_dir
            ),
        ],
    )

    last_ckpt = f"last_{model.model_name}.ckpt"
    resume_ckpt = last_ckpt if os.path.exists(last_ckpt) else None

    trainer.fit(model, data_module, ckpt_path=resume_ckpt)


if __name__ == "__main__":
    snapshot_dir = tempfile.mkdtemp()
    with RsyncSnapshot(snapshot_dir=snapshot_dir):
        main()
