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
from models.base_model import BaseModel
import submitit
import hydra
import http.client as httplib
import datetime
import socket
import git
from pathlib import Path

# "Saves model based on min val loss."""
def setup_best_val_checkpoint(
    model_name: str = "",
    monitor: str = "val_loss",
    mode: str = "min",
    dirpath: Optional[str] = None,
):
    """Saves model based on minimum validation loss

    Args:
        model_name (str, optional): model's name, used for logging. Defaults to "".
        monitor (str, optional): metric used as core differentiator. Defaults to "val_loss".
        mode (str, optional): mode specifying whether to minimize or maximize monitor. Defaults to "min".
        dirpath (Optional[str], optional): path to directory to save model. Defaults to None.

    Returns:
        Model Checkpoint (Pytorch Lightning callback)
    """
    filename = f"best_{monitor}"
    filename += f"_{model_name}"

    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=1,
        # save after training for at least an epoch
        save_on_train_epoch_end=True,
    )


def load_best_val_checkpoint(
    model: BaseModel, monitor="val_loss", datamodule=None
) -> BaseModel:
    ckpt_path = f"best_{monitor}_{model.model_name}.ckpt"
    model = model.load_from_checkpoint(ckpt_path, datamodule=datamodule)
    return model


def load_best_finetuner_val_checkpoint(
    model: BaseModel,
    monitor="val_loss",
    datamodule=None,
) -> BaseModel:
    ckpt_path = f"best_{monitor}_{model.model_name}.ckpt"
    embedding_model = model.embedding_model
    model = model.load_from_checkpoint(
        ckpt_path, embedding_model=embedding_model, datamodule=datamodule
    )
    return model


def setup_last_val_checkpoint(
    dirpath: Optional[str] = None,
    monitor: str = "val_loss",
):
    """Saves last model after validation."""
    return ModelCheckpoint(
        dirpath=dirpath,
        monitor=monitor,
        filename="last_{epoch:02d}-{val_loss:02f}",
        save_on_train_epoch_end=False,
    )


def setup_last_checkpoint(
    dirpath: Optional[str] = None,
    model_name: str = "",
):
    """Saves last model after validation."""
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=f"last_{model_name}",
        save_on_train_epoch_end=True,
    )


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


def create_callbacks(
    config: DictConfig, job_logs_dir: str, model_name: str
) -> List[Callback]:
    """Creates callbacks used during training.
    Sets up callbacks specified in config and additional callbacks for checkpoints.

    Args:
        config: hydra config
        job_logs_dir: directory where job is logged
        model_name: string for name of model used for checkpoint names
    """
    callbacks = initialize_callbacks(config, job_logs_dir=job_logs_dir)
    callbacks.append(setup_last_checkpoint(dirpath=job_logs_dir, model_name=model_name))
    callbacks.append(
        setup_best_val_checkpoint(
            model_name=model_name,
            dirpath=job_logs_dir,
            monitor=config.monitor,
            mode=config.monitor_mode,
        )
    )
    return callbacks


def initialize_callbacks(config: DictConfig, job_logs_dir: str) -> List[Callback]:
    """Initializes callbacks specified in config"""
    callbacks: List[Callback] = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                if "ModelCheckpoint" in cb_conf._target_:
                    # store checkpoints in job_logs_dir
                    callbacks.append(
                        hydra.utils.instantiate(cb_conf, dirpath=job_logs_dir)
                    )
                else:
                    callbacks.append(hydra.utils.instantiate(cb_conf))
    return callbacks


def add_val_combined_accuracy(metrics, train_prop_to_vary: float):
    """Computes val_combined accuracy and loss"""
    if not has_val_accuracy(metrics):
        return metrics

    prefix = ""
    if "online_val_diverse_2d_top_1_accuracy" in metrics:
        prefix = "online_"

    canonical = metrics[f"{prefix}val_canonical_top_1_accuracy"]
    diverse_2d = metrics[f"{prefix}val_diverse_2d_top_1_accuracy"]
    diverse_3d = metrics[f"{prefix}val_diverse_3d_top_1_accuracy"]

    p = train_prop_to_vary
    combined = canonical * (1 - p) + diverse_2d * p / 2.0 + diverse_3d * p / 2.0
    metrics[f"{prefix}val_combined_top_1_accuracy"] = combined
    return metrics


def has_val_accuracy(metrics: dict) -> bool:
    metric_names = [
        "val_diverse_2d_top_1_accuracy",
        "online_val_diverse_2d_top_1_accuracy",
    ]
    for name in metric_names:
        if name in metrics:
            return True
    return False


def load_metrics(model, metrics_ckpt_path=None) -> Dict[str, float]:
    """Returns a dictionary of model metrics.
    If metrics_ckpt_path is passed, metrics are loaded from the checkpoint
    """
    metrics = model.logged_metrics
    if metrics_ckpt_path:
        metrics = torch.load(metrics_ckpt_path)["metrics"]
        metrics = {k: float(v) for k, v in metrics.items()}
    if model.datamodule and hasattr(model.datamodule, "train_prop_to_vary"):
        metrics = add_val_combined_accuracy(
            metrics, model.datamodule.train_prop_to_vary
        )
    return metrics


def log_val_combined_accuracy(metrics, wandb_logger):
    metric_names = ["val_combined_top_1_accuracy", "online_val_combined_top_1_accuracy"]

    for name in metric_names:
        if name in metrics:
            wandb_logger.experiment.log({name: metrics[name]})


def find_existing_checkpoint(dirpath: str) -> Optional[str]:
    """Searches dirpath for an existing model checkpoint.
    If found, returns its path.
    """
    ckpts = list(Path(dirpath).rglob("*.ckpt"))
    if ckpts:
        return str(ckpts[0])
    return None
