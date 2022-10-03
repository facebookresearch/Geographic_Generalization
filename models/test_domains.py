from typing import List, Dict, Optional
import pytorch_lightning as pl
import json
import os
import torch


def test_domains(
    domains: List[str],
    model: pl.LightningModule,
    data_module: pl.LightningDataModule,
    trainer: pl.Trainer,
    save: bool = True,
    folder: str = "",
) -> Dict[str, Optional[float]]:
    """Runs a test loop on each domain.
    Returns: a dictionary {domain: accuracy}
    """
    domain_accuracies = {}
    for domain in domains:
        test_dataloader = data_module.test_dataloader(domain=domain)
        results = trainer.test(model=model, dataloaders=test_dataloader)
        print(f"domain {domain} \n {results=}")
        try:
            accuracy = results[0]
        except KeyError:
            print(f"There was an error saving the accuracy")
            accuracy = None
        domain_accuracies[domain] = accuracy
    if save:
        save_values(domain_accuracies, "domain_accuracies", folder=folder)
    return domain_accuracies


def save_values(values: dict, name: str, folder: str = ""):
    """Saves values as a json with given name"""
    file_path = join_folder(name, folder)
    with open(file_path, "w") as f:
        json.dump(values, f)


def load_values(name: str, folder: str = "") -> dict:
    file_path = join_folder(name, folder)
    with open(file_path, "r") as f:
        values = json.load(f)
    return values


def join_folder(file_name: str, folder: str):
    """Creates a full path for file"""
    if folder:
        return os.path.join(folder, f"{file_name}.json")
    return f"{file_name}.json"


def load_val_accuracy(folder: str, ckpt_name: str) -> float:
    """Loads val accuracy from checkpoint"""
    ckpt_path = os.path.join(folder, ckpt_name)
    print(f"{ckpt_path}")
    val_accuracy = torch.load(ckpt_path)["val_accuracy"].item()
    return val_accuracy


def load_key(folder: str, ckpt_name: str, key: str) -> float:
    """Loads val accuracy from checkpoint"""
    ckpt_path = os.path.join(folder, ckpt_name)
    print(f"{ckpt_path}")
    metric = torch.load(ckpt_path)["metrics"][key].item()
    return metric
