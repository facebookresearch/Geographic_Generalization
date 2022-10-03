from analysis.analyze_runs import Runs
import pandas as pd
from typing import List
import numpy as np


class TableMixin:
    COLUMN_TO_DISPLAY_NAME = {
        "train_canonical_top_1_accuracy": "Train accuracy",
        "train_top_1_accuracy": "Train accuracy",
        "val_canonical_top_1_accuracy": "Held-out accuracy",
    }

    def rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.rename(columns=self.COLUMN_TO_DISPLAY_NAME)
        return df

    def alternate_row_colors(self, latex_table_str) -> str:
        color_str = "\\rowcolors{2}{gray!15}{white}\n"
        return color_str + latex_table_str

    def final_touches(self, latex_table_str) -> str:
        latex_table_str = latex_table_str.replace("model", "")
        return latex_table_str

    @staticmethod
    def add_adjust_box(latex_table_str) -> str:
        latex_table_str = str(latex_table_str)
        latex_table_str = latex_table_str.replace(
            "\\begin{tabular}",
            "\\begin{adjustbox}{width=\\textwidth}\n\\begin{tabular}",
        )
        latex_table_str = latex_table_str.replace(
            "end{tabular}", "end{tabular}\n \\end{adjustbox}"
        )
        return latex_table_str

    def make_gap_columns(
        self, table: pd.DataFrame, factors: List[str], factor_to_display_name: dict
    ) -> pd.DataFrame:
        for factor in factors:
            gaps = (
                table[f"val_diverse_{factor}_top_1_accuracy"]
                - table[f"val_canonical_top_1_accuracy"]
            )
            factor_display_name = factor_to_display_name[factor]
            table[f"{factor_display_name} gap"] = gaps
        return table


class CanonicalTable(TableMixin):
    def __init__(
        self,
        runs: Runs,
        evaluation_type: str = "linear_eval",
        sampling_type: str = "instance",
    ):
        self.runs = runs
        self.evaluation_type = evaluation_type
        self.sampling_type = sampling_type
        self.train_accuracy_name = (
            "train_top_1_accuracy"
            if sampling_type == "image"
            else "train_canonical_top_1_accuracy"
        )

        self.df = self.select_setting_runs()

        self.best_run = runs.find_best_models(self.df)
        self.table = self.make_table()

    def make_table(self) -> pd.DataFrame:
        columns = [
            "model",
            self.train_accuracy_name,
            "val_canonical_top_1_accuracy",
            "val_diverse_Rotation_top_1_accuracy",
            "val_diverse_Background path_top_1_accuracy",
            "val_diverse_Scale_top_1_accuracy",
            "val_diverse_Translation_top_1_accuracy",
            "val_diverse_Spot hue_top_1_accuracy",
        ]
        table = self.best_run[columns]
        display_columns = [
            "model",
            self.train_accuracy_name,
            "val_canonical_top_1_accuracy",
            "Pose gap",
            "Background gap",
            "Size gap",
            "Position gap",
            "Lighting color gap",
        ]
        table = self.make_gap_columns(
            table, self.runs.FACTORS, self.runs.FACTOR_TO_DISPLAY_NAME
        )
        table = table[display_columns]
        table = self.format_table(table)
        table["model"] = table["model"].apply(
            lambda x: self.runs.MODEL_TO_DISPLAY_NAME[x]
        )
        table = self.add_average(table)
        table = self.rename_columns(table)
        return table

    def add_average(self, table):
        table["Average gap"] = table[
            [
                "Pose gap",
                "Background gap",
                "Size gap",
                "Position gap",
                "Lighting color gap",
            ]
        ].mean(axis=1)
        average_row = table.mean()
        average_row["model"] = "Average"
        table = table.append(average_row, ignore_index=True)
        return table

    def format_table(self, table) -> pd.DataFrame:
        table = table.sort_values(by="model")
        table = self.rename_columns(table)
        # multiply by 100
        table[table.select_dtypes(include=["number"]).columns] *= 100
        # table = self.remove_top1_in_names(table)
        return table

    def remove_top1_in_names(self, table):
        new_names = {c: c.replace("_top_1_accuracy", "") for c in table.columns}
        table = table.rename(columns=new_names)
        return table

    def select_setting_runs(self) -> pd.DataFrame:
        if self.evaluation_type == "linear_eval":
            if self.sampling_type == "image":
                return self.runs.canonical_linear_eval
            elif self.sampling_type == "instance":
                return self.runs.canonical_instance_sampling_linear_eval
        elif self.evaluation_type == "finetuning":
            if self.sampling_type == "image":
                return self.runs.canonical_finetuning
            elif self.sampling_type == "instance":
                return self.runs.canonical_instance_sampling_finetuning
        raise ValueError(
            f"{self.evaluation_type} with {self.sampling_type} sampling is not supported"
        )

    def to_latex(self) -> str:
        evalution_type = self.evaluation_type.replace("_", " ")
        # steps = "10k" if self.evaluation_type == "finetuning" else "20k"
        steps = "10k"
        df = self.table
        latex_str = df.to_latex(
            index=False,
            float_format="{:,.2f}".format,
            caption=f"{evalution_type} generalization gaps for each factor: shows results for the best model after {steps} steps of training with adam on 6 log scale learning rates (1e-2 to 1e-6) cross validated on canonical top-1 accuracy for validation images ",
            label=f"{evalution_type}_canonical",
        )
        latex_str = self.add_adjust_box(latex_str)
        latex_str = self.final_touches(latex_str)
        return latex_str


class CompareSupervisionTable(CanonicalTable):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        super().__init__(runs, evaluation_type=evaluation_type)
        self.parent_table = self.table
        self.table = self.make_ssl_comparison(self.parent_table)

    def make_ssl_comparison(self, table):
        table["supervision"] = (
            table["model"]
            .isin(
                [
                    "MLPMixer-1k",
                    "MLPMixer-21k",
                    "ResNet50-1k",
                    "ResNet50-21k",
                    "ViT-1k",
                    "ViT-21k",
                ]
            )
            .apply(lambda x: "supervised" if x else "self-supervised")
        )
        table = table[~table["model"].isin(["CLIPPretrained"])][
            [
                "Pose gap",
                "Background gap",
                "Position gap",
                "Lighting color gap",
                "Size gap",
                "supervision",
            ]
        ]
        return table


class Compare21kTable(CanonicalTable):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        super().__init__(runs, evaluation_type=evaluation_type)
        self.parent_table = self.table
        self.table = self.make_comparison(self.parent_table)

    def make_comparison(self, table):
        table["training_data"] = (
            table["model"]
            .str.contains("21k")
            .apply(lambda x: "ImageNet-21k" if x else "ImageNet")
        )
        table = table[~table["model"].isin(["CLIPPretrained"])][
            [
                "Pose gap",
                "Background gap",
                "Position gap",
                "Lighting color gap",
                "Size gap",
                "training_data",
            ]
        ]
        return table


class SingleFactorTable(TableMixin):
    def __init__(
        self,
        runs: Runs,
        evaluation_type: str = "linear_eval",
        sampling_type: str = "image",
    ):
        self.runs = runs
        self.evaluation_type = evaluation_type
        self.sampling_type = sampling_type
        self.df = self.select_setting_runs()
        self.df["model"] = self.df["model"].apply(
            lambda x: self.runs.MODEL_TO_DISPLAY_NAME[x]
        )

        self.train_accuracy_name = (
            "train_top_1_accuracy"
            if sampling_type == "image"
            else "train_canonical_top_1_accuracy"
        )

        self.columns = [
            "model",
            "factor_to_vary",
            "train_prop_to_vary",
            self.train_accuracy_name,
            "val_canonical_top_1_accuracy",
            "val_diverse_Translation_top_1_accuracy",
            "val_diverse_Rotation_top_1_accuracy",
            "val_diverse_Spot hue_top_1_accuracy",
            "val_diverse_Scale_top_1_accuracy",
            "val_diverse_Background path_top_1_accuracy",
            "Pose gap",
            "Background gap",
            "Position gap",
            "Lighting color gap",
            "Size gap",
        ]

        self.df = self.make_gap_columns(
            self.df, self.runs.FACTORS, self.runs.FACTOR_TO_DISPLAY_NAME
        )
        self.table_raw = self.df[self.columns].sort_values(
            by=["model", "factor_to_vary", "train_prop_to_vary"]
        )
        self.table_raw[self.table_raw.select_dtypes(include=np.number).columns] *= 100.0

        self.table = (
            self.table_raw.pivot_table(
                index=["factor_to_vary", "model"], columns=["train_prop_to_vary"]
            )
        )

    def select_setting_runs(self):
        if self.evaluation_type == "linear_eval":
            if self.sampling_type == "image":
                return self.runs.diverse_single_factor_linear_eval
            return self.runs.diverse_single_factor_instance_sampling_linear_eval
        elif self.evaluation_type == "finetuning":
            if self.sampling_type == "image":
                return self.runs.diverse_single_factor_finetuning
            return self.runs.diverse_single_factor_instance_sampling_finetuning
        raise ValueError(f"{self.evaluation_type} not supported")

    def to_latex(self):
        evalution_type = self.evaluation_type.replace("_", " ")
        return self.table.to_latex(
            index=True,
            float_format="{:,.2%}".format,
            caption=f"{evalution_type} top-1 accuracy: shows performance with varying number of diverse instances seen during training",
            label=f"{evalution_type}_diverse_training",
        )

    def per_factor_to_latex(self):
        evaluation_type = self.evaluation_type.replace("_", " ")
        for factor in self.runs.FACTORS:
            factor_table = self.table.loc[factor][
                [
                    self.train_accuracy_name,
                    "val_canonical_top_1_accuracy",
                    f"val_diverse_{factor}_top_1_accuracy",
                ]
            ]
            latex_str = factor_table.to_latex(
                index=True,
                float_format="{:,.2%}".format,
                caption=f"{factor} diverse {evaluation_type} top-1 accuracy across varying number of diverse training instances",
                label=f"{evaluation_type}_{factor}diverse_training",
            )

            latex_str = str(latex_str)
            latex_str = latex_str.replace(
                "\\begin{tabular}",
                "\\begin{adjustbox}{width=\\textwidth}\n\\begin{tabular}",
            )
            latex_str = latex_str.replace(
                "end{tabular}", "end{tabular}\n \\end{adjustbox}\n"
            )
            print(latex_str)


class SingleFactorPlusCanonicalTable(SingleFactorTable):
    def __init__(
        self,
        runs: Runs,
        evaluation_type: str = "linear_eval",
        sampling_type: str = "image",
    ):
        super().__init__(
            runs, evaluation_type=evaluation_type, sampling_type=sampling_type
        )
        self.original_table = self.table.copy()
        self.canonical_table = self.create_canonical_table()

        self.table = self.combine_tables(self.original_table, self.canonical_table)

    def create_canonical_table(self):
        canonical_table = CanonicalTable(
            self.runs,
            evaluation_type=self.evaluation_type,
        ).table
        canonical_table["train_prop_to_vary"] = 0.0

        aggregate_table = pd.DataFrame()
        for f in self.runs.FACTORS:
            canonical_table["factor_to_vary"] = f
            aggregate_table = aggregate_table.append(
                canonical_table.copy(), ignore_index=True
            )

        aggregate_table = aggregate_table.pivot_table(
            index=["factor_to_vary", "model"], columns=["train_prop_to_vary"]
        )
        return aggregate_table

    def single_factor_table(self, factor: str = "Translation") -> pd.DataFrame:
        display_factor = self.runs.FACTOR_TO_DISPLAY_NAME[factor]
        factor_table = self.table.loc[factor][
            [
                "Held-out canonical accuracy",
                f"{display_factor} gap",
                "Average gap",
            ]
        ].copy()
        factor_table.loc["Average"] = factor_table.mean()
        return factor_table

    def combine_tables(self, canonical, diverse):
        table = canonical.join(diverse).sort_index(axis=1)
        return table

    def single_factor_table(self, factor):
        table = self.single_plus_canonical.table.loc[factor].copy()
        factor_display_name = self.runs.FACTOR_TO_DISPLAY_NAME[factor]
        table = table[factor_display_name + " gap"]
        return table


class CrossFactorGapChanges(SingleFactorPlusCanonicalTable):
    """Compares the gap change between 50% and 0% diversity"""

    def __init__(
        self,
        runs: Runs,
        evaluation_type: str = "linear_eval",
        sampling_type: str = "image",
    ):
        super().__init__(
            runs, evaluation_type=evaluation_type, sampling_type=sampling_type
        )

    def compute_mean_gap_change_for_other_factor(
        self, factor_varying: str = "Translation"
    ) -> np.float:
        gap_changes = []
        for factor in self.runs.FACTORS:
            if factor == factor_varying:
                continue
            factor_display_name = self.runs.FACTOR_TO_DISPLAY_NAME[factor]
            gaps = (
                self.table.groupby("factor_to_vary")
                .mean()
                .loc[factor_varying, f"{factor_display_name} gap"]
            )
            gap_change = gaps.loc[50.0] - gaps.loc[0.0]
            gap_changes.append(gap_change)
        return np.mean(gap_changes)

    def display_cross_factor_gap_changes(self):
        for f in self.runs.FACTORS:
            average_gap_change = self.compute_mean_gap_change_for_other_factor(f)
            print(f, average_gap_change)


class RandomClassGeneralizationTable(TableMixin):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        self.runs = runs
        self.evaluation_type = evaluation_type
        self.df = self.select_setting_runs()

        self.columns = [
            "model",
            "factor_to_vary",
            "train_prop_to_vary",
            "val_canonical_top_1_accuracy",
            "val_class_vary_diverse_Rotation_top_1_accuracy",
            "val_class_vary_diverse_Background path_top_1_accuracy",
            "train_class_vary_diverse_top_1_accuracy",
            "val_diverse_Translation_top_1_accuracy",
            "val_class_vary_canonical_top_1_accuracy",
            "val_diverse_Rotation_top_1_accuracy",
            "train_class_not_vary_canonical_top_1_accuracy",
            "val_diverse_Scale_top_1_accuracy",
            "val_class_not_vary_diverse_Spot hue_top_1_accuracy",
            "val_class_not_vary_diverse_Background path_top_1_accuracy",
            "val_class_not_vary_canonical_top_1_accuracy",
            "val_diverse_Spot hue_top_1_accuracy",
            "val_class_not_vary_diverse_Scale_top_1_accuracy",
            "train_class_vary_canonical_top_1_accuracy",
            "val_class_vary_diverse_Spot hue_top_1_accuracy",
            "val_class_not_vary_diverse_Translation_top_1_accuracy",
            "val_class_not_vary_diverse_Rotation_top_1_accuracy",
            "val_diverse_Background path_top_1_accuracy",
            "val_class_vary_diverse_Translation_top_1_accuracy",
            "val_class_vary_diverse_Scale_top_1_accuracy",
            "diverse_train_class_vary_canonical_Translation_top_1_accuracy",
            "diverse_train_class_not_vary_Scale_top_1_accuracy",
            "diverse_train_class_vary_canonical_Scale_top_1_accuracy",
            "test_diverse_Background path_top_1_accuracy",
            "test_diverse_Rotation_top_1_accuracy",
            "diverse_train_class_vary_canonical_Spot hue_top_1_accuracy",
            "diverse_train_class_vary_canonical_Background path_top_1_accuracy",
            "diverse_train_class_not_vary_Translation_top_1_accuracy",
            "test_diverse_Translation_top_1_accuracy",
            "test_diverse_Scale_top_1_accuracy",
            "diverse_train_class_not_vary_Background path_top_1_accuracy",
            "test_canonical_top_1_accuracy",
            "diverse_train_class_not_vary_Rotation_top_1_accuracy",
            "test_diverse_Spot hue_top_1_accuracy",
            "diverse_train_class_vary_canonical_Rotation_top_1_accuracy",
            "diverse_train_class_not_vary_Spot hue_top_1_accuracy",
            "diverse_train_canonical_Rotation_top_1_accuracy",
            "diverse_train_canonical_Scale_top_1_accuracy",
            "diverse_train_canonical_Translation_top_1_accuracy",
            "diverse_train_canonical_Spot hue_top_1_accuracy",
            "diverse_train_canonical_Background path_top_1_accuracy",
            "train_canonical_top_1_accuracy",
            "train_diverse_top_1_accuracy",
        ]
        self.df = self.df[self.columns].sort_values(
            by=["model", "factor_to_vary", "train_prop_to_vary"]
        )

        self.table = self.make_table()

    def select_setting_runs(self):
        if self.evaluation_type == "linear_eval":
            return self.runs.random_class_generalization_linear_eval
        elif self.evaluation_type == "finetuning":
            return self.runs.random_class_generalization_finetuning
        raise ValueError(f"{self.evaluation_type} not supported")

    def make_factor_table(self, factor: str = "Rotation") -> pd.DataFrame:
        df = self.df[self.df["factor_to_vary"] == factor]
        # (val_class_vary - val_class_not_vary)
        df[f"{factor}_gap"] = df[
            f"val_class_not_vary_diverse_{factor}_top_1_accuracy"
        ].subtract(df[f"val_class_vary_diverse_{factor}_top_1_accuracy"])
        columns = [
            "model",
            f"{factor}_gap",
            "train_class_not_vary_canonical_top_1_accuracy",
            "val_canonical_top_1_accuracy",
            f"val_class_vary_diverse_{factor}_top_1_accuracy",
            f"val_class_not_vary_diverse_{factor}_top_1_accuracy",
        ]
        return df[columns]

    def add_average(self, table):
        table["Average gap"] = table[
            [
                "Pose gap",
                "Background gap",
                "Size gap",
                "Position gap",
                "Lighting color gap",
            ]
        ].mean(axis=1)
        average_row = table.mean()
        average_row["model"] = "Average"
        table = table.append(average_row, ignore_index=True)
        return table

    def make_table(self) -> pd.DataFrame:
        """Summarizes the class generalization gaps"""
        df = None
        for factor in self.runs.FACTORS:
            columns = ["model", f"{factor}_gap"]
            factor_df = self.make_factor_table(factor=factor)[columns]
            if df is None:
                df = factor_df
            else:
                df = df.merge(factor_df, on="model")
        factor_gap_to_display_name = {
            f"{f}_gap": f"{self.runs.FACTOR_TO_DISPLAY_NAME[f]} gap"
            for f in self.runs.FACTORS
        }
        df = df.rename(columns=factor_gap_to_display_name)
        df["model"] = df["model"].apply(lambda x: self.runs.MODEL_TO_DISPLAY_NAME[x])
        df = self.add_average(df)
        return df

    def to_latex(self):
        evalution_type = self.evaluation_type.replace("_", " ")
        latex_str = self.table.to_latex(
            index=False,
            float_format="{:,.2f}".format,
            caption=f"{evalution_type} class generalization top-1 accuracy gaps: shows validation top-1 accuracy difference between classes (27 randomly selected) seen with diversity and those not.",
            label=f"{evalution_type}_class_generalization",
        )
        latex_str = self.add_adjust_box(latex_str)
        return latex_str


class PairedFactorTable(TableMixin):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        self.runs = runs
        self.evaluation_type = evaluation_type
        self.df = self.select_setting_runs()
        self.df = self.convert_factors_to_vary_to_str(self.df)
        self.table = self.make_table()

    def convert_factors_to_vary_to_str(self, df):
        df["factors_to_vary_original_list"] = df["factors_to_vary"]
        df["factors_to_vary"] = df["factors_to_vary"].apply(lambda x: ", ".join(x))
        return df

    def select_setting_runs(self):
        if self.evaluation_type == "linear_eval":
            return self.runs.diverse_paired_factors_linear_eval
        elif self.evaluation_type == "finetuning":
            return self.runs.diverse_paired_factors_finetuning
        raise ValueError(f"{self.evaluation_type} not supported")

    def make_table(self) -> pd.DataFrame:
        df = self.df
        df = self.make_gap_columns(
            df, self.runs.FACTORS, self.runs.FACTOR_TO_DISPLAY_NAME
        )
        columns = [
            "factors_to_vary",
            "model",
            "train_canonical_top_1_accuracy",
            "val_canonical_top_1_accuracy",
        ]
        columns += [
            f"{self.runs.FACTOR_TO_DISPLAY_NAME[f]} gap" for f in self.runs.FACTORS
        ]
        df = df[columns].sort_values(by=["factors_to_vary", "model"])
        # df = df.set_index(["factors_to_vary"])
        return df


class AllFactorTable(TableMixin):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        self.runs = runs
        self.evaluation_type = evaluation_type
        self.df = self.select_setting_runs()
        self.table = self.make_table()

    def select_setting_runs(self):
        if self.evaluation_type == "linear_eval":
            return self.runs.diverse_all_factors_linear_eval
        elif self.evaluation_type == "finetuning":
            return self.runs.diverse_all_factors_linear_eval
        raise ValueError(f"{self.evaluation_type} not supported")

    def make_table(self) -> pd.DataFrame:
        df = self.df
        df = self.make_gap_columns(
            df, self.runs.FACTORS, self.runs.FACTOR_TO_DISPLAY_NAME
        )
        columns = [
            "model",
            "train_canonical_top_1_accuracy",
            "val_canonical_top_1_accuracy",
        ]
        columns += [
            f"{self.runs.FACTOR_TO_DISPLAY_NAME[f]} gap" for f in self.runs.FACTORS
        ]
        df = df[columns].sort_values(by="model")
        df["model"] = df["model"].apply(lambda x: self.runs.MODEL_TO_DISPLAY_NAME[x])
        # df = df.set_index(["factors_to_vary"])
        return df
