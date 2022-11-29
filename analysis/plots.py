from analysis import analyze_runs
from analysis.analyze_runs import Runs
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from analysis import tables
import numpy as np
import os
from typing import List, Optional, Dict


class PlotMixin:
    COLORS = px.colors.qualitative.Plotly
    BASE_MODELS = ["ViT", "CLIP", "ResNet50", "MAE", "iBot", "SimCLR", "MLPMixer"]
    BASE_MODEL_TO_COLOR = dict(zip(BASE_MODELS, COLORS[: len(BASE_MODELS)]))

    def stylize_pot(self, fig: go.Figure) -> go.Figure:
        fig.update_layout(template="plotly_white")
        return fig

    def get_base_model(self, model: str) -> str:
        model = model.replace("Pretrained", "")
        model = model.replace("-", "")
        model = model.replace("21k", "")
        model = model.replace("1k", "")
        return model

class AllFactorPlot(PlotMixin):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        self.all_runs = runs
        self.evaluation_type = evaluation_type

        self.table = tables.AllFactorTable(runs, evaluation_type=evaluation_type).table

        self.models = self.table["model"].unique().tolist()
        self.factor_display_names = [
            self.all_runs.FACTOR_TO_DISPLAY_NAME[f] for f in self.all_runs.FACTORS
        ]

    def plot(
        self,
    ) -> go.Figure:
        """Bar plots measuring the gaps of factors when all are seen with diversity"""
        fig = go.Figure()
        df = self.table
        gap_names = [f"{f} gap" for f in self.factor_display_names]

        for model in self.models:
            model_df = df[df["model"] == model]
            base_model = self.get_base_model(model)
            fig.add_trace(
                go.Bar(
                    x=model_df[gap_names].values[0] * 100,
                    y=gap_names,
                    name=model,
                    orientation="h",
                    marker_pattern_shape="/" if "21k" in model else "",
                    marker_color=self.BASE_MODEL_TO_COLOR[base_model],
                )
            )
        fig.update_xaxes(
            title_text=f"top-1 accuracy drop when all factors vary <br> ({self.evaluation_type.replace('_', ' ')})"
        )
        fig.update_layout(barmode="group")
        fig = self.stylize_pot(fig)
        return fig

    def save(self, save_dir: str = "plots/"):
        fig = self.plot()
        save_path = os.path.join(
            save_dir,
            f"all_factors_{self.evaluation_type}_gaps.pdf",
        )
        fig.write_image(save_path, format="pdf")



if __name__ == "__main__":
    # run via: python analysis/plots.py
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(dir_path, "plots/")

    runs = Runs()
