from functools import cache
import pandas as pd
import wandb
import spacy
import numpy as np
import glob
from pathlib import Path

from datasets.shapenet.shapenet_extended import (
    ShapeNetExtended,
    ShapeNetRandomClassSplitDataModule,
    SingleFovHandler,
)


class Runs:
    def __init__(
        self,
        entity: str,
        project: str,
        exclude_zero_shot: bool = True,
    ):
        self.entity = entity
        self.project = project

        api = wandb.Api()
        self.runs = api.runs(entity + "/" + project)

        self.nested_config_keys = ["module", "evaluation_module", "datamodule"]
        self.config_keys = ["name", "job_logs_dir", "model_name"]

        self.df = self.create_df()

    def create_df(self):
        """Creates a dataframe of all runs"""
        data = []
        for run in self.runs:
            run_data = dict()
            run_data.update(self.extract_summary(run.summary._json_dict))
            run_data.update(self.extract_config(run.config))
            run_data.update({"tags": run.tags})
            if "sanity_check" in run.tags:
                run_data.update({"is_sanity_check": True})
            else:
                run_data.update({"is_sanity_check": False})
            data.append(run_data)

        runs_df = pd.DataFrame.from_records(data)
        # filter sanity checks
        runs_df = runs_df[~runs_df["is_sanity_check"]]
        return runs_df

    def extract_config(self, config: dict) -> dict:
        data = dict()
        for config_key in self.config_keys:
            if config_key in config:
                data[config_key] = config[config_key]

        # data module keys
        for module in self.nested_config_keys:
            if module not in config:
                continue
            for k in config[module]:
                if k == "_target_":
                    data[module] = config[module]["_target_"]
                else:
                    data[k] = config[module][k]

        data["model"] = config["module"]["_target_"].split(".")[-1]

        return data

    def extract_summary(self, summary: dict) -> dict:
        """Summary containing accuracy and other metrics"""
        data = dict()
        for (
            k,
            v,
        ) in summary.items():
            if k.startswith("_"):
                continue
            # skip class specific top1
            if k.split("_")[-1].isdigit():
                continue
            data[k] = v
        return data
