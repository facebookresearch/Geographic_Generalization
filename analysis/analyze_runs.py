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
    FACTORS = ["Translation", "Rotation", "Spot hue", "Scale", "Background path"]
    FACTOR_TO_DISPLAY_NAME = {
        "Translation": "Position",
        "Rotation": "Pose",
        "Spot hue": "Lighting color",
        "Background path": "Background",
        "Scale": "Size",
    }
    MODEL_TO_DISPLAY_NAME = {
        "ResNet50Pretrained1k": "ResNet50-1k",
        "ResNet50Pretrained21k": "ResNet50-21k",
        "CLIPPretrained": "CLIP",
        "SimCLRPretrained": "SimCLR",
        "MAEPretrained": "MAE",
        "iBotPretrained21k": "iBot-21k",
        "iBotPretrained1k": "iBot-1k",
        "MLPMixerPretrained21k": "MLPMixer21k",
        "MLPMixerPretrained1k": "MLPMixer1k",
        "ViTPretrained1k": "ViT-1k",
        "ViTPretrained21k": "ViT-21k",
    }

    def __init__(
        self,
        entity="robustness-limits",
        project="robustness-limits",
        exclude_zero_shot: bool = True,
    ):
        self.entity = entity
        self.project = project

        api = wandb.Api()
        self.runs = api.runs(entity + "/" + project)

        self.nested_config_keys = ["module", "evaluation_module", "datamodule"]
        self.config_keys = ["name", "job_logs_dir", "model_name"]
        self.exclude_zero_shot = exclude_zero_shot

        self.df = self.create_df()

    @property
    def FACTOR_DISPLAY_NAMES(self):
        return [self.FACTOR_TO_DISPLAY_NAME[f] for f in self.FACTORS]

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
        # exclude VICReg
        runs_df = runs_df[~runs_df["model"].str.contains("VICRegPretrained")]
        if self.exclude_zero_shot:
            # exclude zero_shot
            runs_df = runs_df[~runs_df["model"].str.contains("zero_shot")]
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

    def find_best_models(self, df, metric="val_canonical_top_1_accuracy"):
        df = df.loc[df.groupby("model")[metric].idxmax()]
        return df

    @property
    def canonical_linear_eval(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetv2SingleFactorDataModule")]
        df = df[df["train_prop_to_vary"] == 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.LinearEval"]
        df = df[~df["train_top_1_accuracy"].isna()]
        return df

    @property
    def canonical_finetuning(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetv2SingleFactorDataModule")]
        df = df[df["train_prop_to_vary"] == 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.Finetuning"]
        df = df[~df["train_top_1_accuracy"].isna()]
        return df

    @property
    def canonical_instance_sampling_linear_eval(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetIndividualFactorDataModule")]
        df = df[df["train_prop_to_vary"] == 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.LinearEval"]
        df = df[~df["train_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def canonical_instance_sampling_finetuning(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetIndividualFactorDataModule")]
        df = df[df["train_prop_to_vary"] == 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.Finetuning"]
        df = df[~df["train_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def diverse_single_factor_linear_eval(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetv2SingleFactorDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.LinearEval"]
        # filter requeued jobs
        df = df[~df["train_top_1_accuracy"].isna()]
        return df

    @property
    def diverse_single_factor_finetuning(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetv2SingleFactorDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.Finetuning"]
        # filter requeued jobs
        df = df[~df["train_top_1_accuracy"].isna()]
        return df

    @property
    def diverse_single_factor_instance_sampling_linear_eval(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetIndividualFactorDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.LinearEval"]
        # filter requeued jobs
        df = df[~df["train_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def diverse_single_factor_instance_sampling_finetuning(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetIndividualFactorDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.Finetuning"]
        # filter requeued jobs
        df = df[~df["train_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def diverse_paired_factors_linear_eval(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetPairedFactorDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.LinearEval"]
        # filter requeued jobs
        df = df[~df["train_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def diverse_paired_factors_finetuning(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetPairedFactorDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.Finetuning"]
        # filter requeued jobs
        df = df[~df["train_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def diverse_all_factors_linear_eval(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetAllFactorsDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.LinearEval"]
        # filter requeued jobs
        df = df[~df["train_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def diverse_all_factors_finetuning(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetAllFactorsDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.Finetuning"]
        # filter requeued jobs
        df = df[~df["train_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def random_class_generalization_linear_eval(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetRandomClassSplitDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.LinearEval"]
        # filter requeued jobs
        df = df[~df["train_class_vary_canonical_top_1_accuracy"].isna()]
        return df

    @property
    def random_class_generalization_finetuning(self):
        df = self.df
        df = df[df["datamodule"].str.endswith("ShapeNetRandomClassSplitDataModule")]
        df = df[df["train_prop_to_vary"] > 0.0]
        df = df[df["evaluation_module"] == "models.evaluation.Finetuning"]
        # filter requeued jobs
        df = df[~df["train_class_vary_canonical_top_1_accuracy"].isna()]
        return df


class ClassSimilarity:
    def __init__(
        self,
        predictions_dir: str = "/checkpoint/marksibrahim/results/robustness-limits/predictions/",
    ):
        self.predictions_dir = predictions_dir
        self.df = self.read_predictions()
        self.synset_to_name = {
            "02691156": "airplane;aeroplane;plane",
            "03759954": "microphone;mike",
            "03797390": "mug",
            "04460130": "tower",
            "02942699": "camera;photographic camera",
            "04090263": "rifle",
            "03991062": "pot;flowerpot",
            "02801938": "basket;handbasket",
            "02880940": "bowl",
            "02828884": "bench",
            "04330267": "stove",
            "02958343": "car;auto;automobile;machine;motorcar",
            "03207941": "dishwasher;dish washer;dishwashing machine",
            "03325088": "faucet;spigot",
            "03513137": "helmet",
            "03642806": "laptop;laptop computer",
            "03928116": "piano;pianoforte;forte-piano",
            "03593526": "jar",
            "03001627": "chair",
            "02843684": "birdhouse",
            "04099429": "rocket;projectile",
            "03761084": "microwave;microwave oven",
            "03337140": "file;file cabinet;filing cabinet",
            "02946921": "can;tin;tin can",
            "04468005": "train;railroad train",
            "02808440": "bathtub;bathing tub;bath;tub",
            "04004475": "printer;printing machine",
            "03691459": "loudspeaker;speaker;speaker unit;loudspeaker system;speaker system",
            "03211117": "display;video display",
            "02924116": "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger vehi",
            "04379243": "table",
            "02747177": "ashcan;trash can;garbage can;wastebin;ash bin;ash-bin;ashbin;dustbin;trash barrel;trash bin",
            "03938244": "pillow",
            "03624134": "knife",
            "02871439": "bookshelf",
            "03046257": "clock",
            "04225987": "skateboard",
            "03790512": "motorcycle;bike",
            "03467517": "guitar",
            "02954340": "cap",
            "04074963": "remote control;remote",
            "02818832": "bed",
            "03710193": "mailbox;letter box",
            "03261776": "earphone;earpiece;headphone;phone",
            "04530566": "vessel;watercraft",
            "02773838": "bag;traveling bag;travelling bag;grip;suitcase",
            "04401088": "telephone;phone;telephone set",
            "02933112": "cabinet",
            "03948459": "pistol;handgun;side arm;shooting iron",
            "03636649": "lamp",
            "02876657": "bottle",
            "03085013": "computer keyboard;keypad",
            "04256520": "sofa;couch;lounge",
            "04554684": "washer;automatic washer;washing machine",
        }
        self.synset_to_class_idx = self.load_class_idx_to_synset()
        self.class_idx_to_synset = {v: k for k, v in self.synset_to_class_idx.items()}

        self.varying_synsets = list(
            ShapeNetRandomClassSplitDataModule.synset_to_name_subset.keys()
        )
        self.varying_class_idx = [
            self.synset_to_class_idx[s] for s in self.varying_synsets
        ]

        self.nlp = spacy.load("en_core_web_lg")

        # true_class (idx), loader_name, accuracy
        # loader_names: val_canonical, val_diverse_Background path etc.
        self.accuracies_df = self.compute_class_accuracies()
        self.accuracies_df = self.add_class_name(self.accuracies_df)
        self.accuracies_df = self.add_similarity(self.accuracies_df)
        self.accuracies_df = self.add_accuracy_gaps(self.accuracies_df)

    def read_predictions(self) -> pd.DataFrame:
        predictions_dir_glob = self.predictions_dir + "*.csv"

        df = pd.DataFrame()

        for predictions_path in glob.glob(predictions_dir_glob):
            model_df = pd.read_csv(predictions_path)
            model_name = Path(predictions_path).stem.replace("_predictions", "")
            model_df["model"] = model_name
            df = df.append(model_df)
        return df

    def load_class_idx_to_synset(self):
        data_dir = "/checkpoint/garridoq/datasets_fov/individual"
        fov_handler = SingleFovHandler({})
        ds = ShapeNetExtended(data_dir, fov_handler)
        return ds.class_to_idx

    @cache
    def compute_class_name_pair_similarity(
        self, class_name1: str, class_name2: str
    ) -> float:
        class_name1 = class_name1.split(";")[0]
        class_name2 = class_name2.split(";")[0]
        similarity = self.nlp(class_name1).similarity(self.nlp(class_name2))
        return similarity

    @cache
    def compute_class_idx_pair_similarity(
        self, class1_idx: int, class2_idx: int
    ) -> float:
        class1_name = self.synset_to_name[self.class_idx_to_synset[class1_idx]]
        class2_name = self.synset_to_name[self.class_idx_to_synset[class2_idx]]
        similarity = self.compute_class_name_pair_similarity(class1_name, class2_name)
        return similarity

    def compute_class_accuracies(self):
        df = self.df.copy()
        df["is_correct"] = df["pred"] == df["true_class"]

        correct = df.groupby(["true_class", "loader_name", "model"]).sum()["is_correct"]
        total = df.groupby(["true_class", "loader_name", "model"]).count()["is_correct"]

        accuracies = correct / total
        accuracies = accuracies.reset_index()
        accuracies = accuracies.rename(columns={"is_correct": "accuracy"})
        return accuracies

    def add_class_name(self, table):
        table["class_name"] = table["true_class"].apply(
            lambda x: self.synset_to_name[self.class_idx_to_synset[x]]
        )
        return table

    def add_similarity(self, table):
        """Adds similarity to nearest class seen varying"""
        similarities = []
        for class_idx in table["true_class"].values:
            similarity_to_varying = [
                self.compute_class_idx_pair_similarity(class_idx, idx)
                for idx in self.varying_class_idx
            ]
            similarities.append(np.max(similarity_to_varying))

        table["similarity"] = similarities
        return table

    def add_accuracy_gaps(self, table):
        canonical_df = table[table["loader_name"] == f"val_canonical"]
        model_canonical_df = canonical_df.groupby("model").mean()["accuracy"]

        table = table.set_index(["model", "true_class", "loader_name"])
        table["accuracy gap"] = table["accuracy"] - model_canonical_df
        table = table.reset_index()
        return table
