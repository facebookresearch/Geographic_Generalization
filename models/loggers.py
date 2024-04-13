"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
import torch
import wandb
from typing import List, Optional, Tuple, Dict
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import logging
from omegaconf import OmegaConf, DictConfig
import yaml
import os
import random
from models.classifier_model import ClassifierModule
import submitit
import hydra
import http.client as httplib
import datetime
import socket
import git
from pathlib import Path


def setup_wandb(
    config: DictConfig, log: logging.Logger, git_hash: str = ""
) -> WandbLogger:
    log_job_info(log)
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))
    job_logs_dir = os.getcwd()
    # increase timeout per wandb folks' suggestion
    os.environ["WANDB_INIT_TIMEOUT"] = "60"
    config_dict["job_logs_dir"] = job_logs_dir
    config_dict["git_hash"] = git_hash

    try:
        wandb_logger = WandbLogger(
            config=config_dict,
            settings={"start_method": "fork"},
            **config.wandb,
        )
    except Exception as e:
        print(f"exception: {e}")
        print(log_internet_status())
        print("starting wandb in offline mode. To sync logs run")
        print(f"wandb sync {job_logs_dir}")
        os.environ["WANDB_MODE"] = "offline"
        wandb_logger = WandbLogger(
            config=config_dict,
            settings={"start_method": "fork"},
            **config.wandb,
        )
    return wandb_logger


def get_git_hash() -> Optional[str]:
    try:
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        return sha
    except:
        print("not able to find git hash")


def check_internet():
    conn = httplib.HTTPSConnection("8.8.8.8", timeout=5)
    try:
        conn.request("HEAD", "/")
        return True
    except Exception:
        return False
    finally:
        conn.close()


def log_internet_status() -> str:
    have_internet = check_internet()
    if have_internet:
        return "successfully connected to Google"
    time = datetime.datetime.now()
    machine_name = socket.gethostname()
    return f"Could not connect to Google at {time} from {machine_name}"


def log_info_debugging():
    machine_name = socket.gethostname()
    print("running on", machine_name)
    print("setting NCCL_DEBUG=INFO to show DDP errors")
    os.environ["NCCL_DEBUG"] = "INFO"
    # set to DETAIL for runtime logging.
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # can help hanging DDP runs by enforcing out of sync process to wait
    # this can impact performance by +10%
    os.environ["NCCL_BLOCKING_WAIT"] = "1"
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
    os.environ["NCCL_SHM_DISABLE"] = "1"


def log_job_info(log: logging.Logger):
    """Logs info about the job directory and SLURM job id"""
    job_logs_dir = os.getcwd()
    log.info(f"Logging to {job_logs_dir}")
    job_id = "local"

    try:
        job_env = submitit.JobEnvironment()
        job_id = job_env.job_id
    except RuntimeError:
        pass

    log.info(f"job id {job_id}")


@rank_zero_only
def print_config(
    config: DictConfig,
    resolve: bool = True,
) -> None:
    """Saves and prints content of DictConfig
    Args:
        config (DictConfig): Configuration composed by Hydra.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    run_configs = OmegaConf.to_yaml(config, resolve=resolve)
    # try:
    #     git_hash = git
    #     run_configs =
    # except:
    #     print("not storing git hash")
    #     pass
    print(run_configs)
    with open("run_configs.yaml", "w") as f:
        OmegaConf.save(config=config, f=f)


def find_existing_checkpoint(dirpath: str) -> Optional[str]:
    """Searches dirpath for an existing model checkpoint.
    If found, returns its path.
    """
    ckpts = list(Path(dirpath).rglob("*.ckpt"))
    if ckpts:
        return str(ckpts[0])
    return None
