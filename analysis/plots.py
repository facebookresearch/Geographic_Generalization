import analysis
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
from sklearn.linear_model import LinearRegression
from scipy.stats import binned_statistic
from scipy import stats


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


class DataDiversityPlot(PlotMixin):
    def __init__(
        self,
        runs: Runs,
        evaluation_type: str = "linear_eval",
        sampling_type: str = "image",
    ):
        self.all_runs = runs
        self.evaluation_type = evaluation_type
        self.sampling_type = sampling_type

        self.table = tables.SingleFactorTable(
            runs, evaluation_type=evaluation_type, sampling_type=sampling_type
        ).table_raw
        self.models = self.table["model"].unique()

    def single_factor_scatter_helper(
        self,
        factor,
        fig: go.Figure,
        factor_affected: str,
        showlegend: bool = True,
        row=None,
        col=None,
    ):
        factor_affected_display_name = self.all_runs.FACTOR_TO_DISPLAY_NAME[
            factor_affected
        ]

        df = self.table[self.table["factor_to_vary"] == factor]

        for model in self.models:
            model_df = df[df["model"] == model]
            fig.add_trace(
                go.Scatter(
                    x=model_df["train_prop_to_vary"],
                    y=model_df[f"{factor_affected_display_name} gap"],
                    mode="lines+markers",
                    name=model,
                    line=self.get_model_line_style(model),
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )

        x, y_hat, slope = self.compute_line_of_best_fit(
            factor, factor_affected_display_name
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_hat,
                mode="lines",
                name="best fit",
                line=dict(color="gray"),
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
        fig.add_annotation(
            text=f"slope={float(slope):.2f}",
            row=row,
            col=col,
            # The arrow head will be 25% along the x axis, starting from the left
            x=30.0,
            # The arrow head will be 40% along the y axis, starting from the bottom
            y=5.0,
            showarrow=False,
        )

        return fig

    def compute_line_of_best_fit(self, factor, factor_affected_display_name):
        df = self.table[self.table["factor_to_vary"] == factor]
        x = df.train_prop_to_vary.values.reshape(-1, 1)
        y = df[f"{factor_affected_display_name} gap"].values.reshape(-1, 1)
        linear_model = LinearRegression()
        linear_model.fit(np.array(x).reshape(-1, 1), y)
        slope = linear_model.coef_[0]
        x_range = np.arange(0, 100, 1)
        y_hat = np.squeeze(linear_model.predict(x_range.reshape(-1, 1)))
        return x_range, y_hat, slope

    def plot_single_factor(
        self, factor: str = "Translation", factor_affected: str = None
    ) -> go.Figure:
        """Shows the affect of varying factor on factor_affected.
        If factor_affected is none then factor_affected=factor
        """
        if factor_affected is None:
            factor_affected = factor

        fig = go.Figure()
        fig = self.single_factor_scatter_helper(factor, fig, factor_affected)
        factor_display_name = self.all_runs.FACTOR_TO_DISPLAY_NAME[factor]
        factor_affected_display_name = factor_affected
        fig.update_layout(
            title=f"{factor_display_name} generalization gap",
            xaxis_title=f"percent of diverse {factor_display_name}training instances",
            yaxis_title=f"top-1 accuracy drop for {factor_affected_display_name}",
        )
        fig = self.stylize_pot(fig)
        return fig

    def get_model_line_style(self, model: str) -> dict:
        """Line style dictionary for plot"""
        base_model = self.get_base_model(model)
        color = self.BASE_MODEL_TO_COLOR[base_model]
        dash = "dash" if "21k" in model else "solid"
        line_style = dict(color=color, dash=dash)
        return line_style

    def plot(self, num_cols=3) -> go.Figure:
        num_figures = 5
        num_rows = -(num_figures // -num_cols)
        subplot_titles = [
            self.all_runs.FACTOR_TO_DISPLAY_NAME[f] for f in self.all_runs.FACTORS
        ]

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            shared_yaxes=True,
        )

        for i, factor in enumerate(self.all_runs.FACTORS):
            showlegend = True if i == 0 else False
            row = i // num_cols
            col = i % num_cols
            factor_affected = factor
            fig = self.single_factor_scatter_helper(
                factor,
                fig,
                factor_affected,
                showlegend=showlegend,
                row=row + 1,
                col=col + 1,
            )
        fig.update_yaxes(title_text="Top-1 accuracy drop", row=1, col=1)
        fig.update_yaxes(title_text="Top-1 accuracy drop", row=2, col=1)
        fig.update_xaxes(title_text="Percent of varying training instances")
        fig = self.stylize_pot(fig)
        fig.update_layout(legend=dict(yanchor="top", y=0.45, xanchor="left", x=0.81))
        return fig

    def plot_one_row(self, num_cols=6) -> go.Figure:
        subplot_titles = [
            self.all_runs.FACTOR_TO_DISPLAY_NAME[f] for f in self.all_runs.FACTORS
        ]

        fig = make_subplots(
            rows=1,
            cols=5,
            subplot_titles=subplot_titles,
            shared_yaxes=True,
        )

        for col, factor in enumerate(self.all_runs.FACTORS):
            showlegend = True if col == 4 else False
            factor_affected = factor
            fig = self.single_factor_scatter_helper(
                factor,
                fig,
                factor_affected,
                showlegend=showlegend,
                row=1,
                col=col + 1,
            )
            if col != 2:
                fig.update_xaxes(title_text="", row=1, col=col + 1)
            else:
                if self.sampling_type == "image":
                    title = "Percent of varying images across all instances"
                elif self.sampling_type == "instance":
                    title = "Perecent of varying training samples"

                fig.update_xaxes(title_text=title, row=1, col=col + 1)
        # fig.update_yaxes(title_text="top-1 accuracy drop", row=1, col=1)
        # fig.update_yaxes(title_text="top-1 accuracy drop", row=2, col=1)
        fig = self.stylize_pot(fig)
        fig.update_layout(width=1200, height=320)
        fig.update_layout(legend=dict(yanchor="bottom", y=-0.8, orientation="h"))
        return fig

    def plot_factor_effects(self, factor_to_vary, num_cols=3) -> go.Figure:
        num_figures = 5
        num_rows = -(num_figures // -num_cols)
        subplot_titles = [
            self.all_runs.FACTOR_TO_DISPLAY_NAME[f] for f in self.all_runs.FACTORS
        ]

        fig = make_subplots(
            rows=num_rows,
            cols=num_cols,
            subplot_titles=subplot_titles,
            shared_yaxes=True,
        )

        for i, factor_affected in enumerate(self.all_runs.FACTORS):
            showlegend = True if i == 0 else False
            row = i // num_cols
            col = i % num_cols
            fig = self.single_factor_scatter_helper(
                factor_to_vary,
                fig,
                factor_affected,
                showlegend=showlegend,
                row=row + 1,
                col=col + 1,
            )
        factor_to_vary_display = self.all_runs.FACTOR_TO_DISPLAY_NAME[factor_to_vary]

        fig.update_yaxes(title_text="top-1 accuracy drop", row=1, col=1)
        fig.update_yaxes(title_text="top-1 accuracy drop", row=2, col=1)
        fig.update_xaxes(
            title_text=f"Percent of diverse {factor_to_vary_display} training instances"
        )
        fig = self.stylize_pot(fig)
        fig.update_layout(
            legend=dict(yanchor="top", y=0.45, xanchor="left", x=0.81),
            title=f"Generalization when {factor_to_vary_display} varies in training",
        )
        return fig

    def save_appendix(self, save_dir: str = "plots/"):
        for factor_to_vary in self.all_runs.FACTORS:
            fig = self.plot_factor_effects(factor_to_vary)
            save_path = os.path.join(
                save_dir,
                f"appendix_{factor_to_vary}_{self.evaluation_type}_diverse_gaps.pdf",
            )
            fig.update_layout(width=1200, height=520)
            fig.write_image(save_path, format="pdf")

    def save(self, save_dir: str = "plots/"):
        # fig = self.plot()
        fig = self.plot_one_row()
        save_path = os.path.join(save_dir, f"{self.evaluation_type}_diverse_gaps.pdf")
        # fig.update_layout(
        #     margin=dict(t=30),
        # )
        fig.write_image(save_path, format="pdf")
        return fig


class SpillOverDiversityHeatmap(DataDiversityPlot):
    def __init__(
        self,
        runs: Runs,
        evaluation_type: str = "linear_eval",
        sampling_type: str = "instance",
    ):
        super().__init__(
            runs, evaluation_type=evaluation_type, sampling_type=sampling_type
        )

        self.gap_names = [f"{f} gap" for f in self.all_runs.FACTOR_DISPLAY_NAMES]

        self.slope_table = self.make_slope_table()
        self.normalized_slope_table = self.make_normalized_table()

    def make_slope_table(self) -> pd.DataFrame:
        records = []
        for factor in self.all_runs.FACTORS:
            records.append(self.compute_slopes(factor))
        table = pd.DataFrame.from_records(records)
        cols = ["factor_to_vary"] + self.gap_names
        table = table[cols]
        table = table.rename(columns={"factor_to_vary": "varying factor"})
        return table

    def make_normalized_table(self) -> pd.DataFrame:
        table = self.slope_table
        diag_values = np.diag(table[self.gap_names])
        normalized_table = table.copy()
        normalized_table[self.gap_names] = normalized_table[self.gap_names].divide(
            diag_values
        )
        return normalized_table

    def compute_slopes(self, factor: str = "Translation") -> Dict:
        average_gaps = (
            self.table[self.table["factor_to_vary"] == factor]
            .groupby("train_prop_to_vary")
            .mean()
        )

        gap_names = [f"{f} gap" for f in self.all_runs.FACTOR_DISPLAY_NAMES]

        factor_to_slope = dict()

        for gap_name in gap_names:
            linear_fit = LinearRegression()
            slope = linear_fit.fit(
                average_gaps[gap_name].index.values.reshape(-1, 1),
                average_gaps[gap_name].values.reshape(-1, 1),
            ).coef_[0][0]
            factor_to_slope[gap_name] = slope * 100

        factor_to_slope["factor_to_vary"] = self.all_runs.FACTOR_TO_DISPLAY_NAME[factor]
        return factor_to_slope

    def plot_table(self):
        table = self.slope_table
        return table.style.background_gradient(cmap="Blues")

    def plot(self, normalized: bool = False) -> go.Figure:
        table = self.normalized_slope_table if normalized else self.slope_table
        table = table.copy()
        df = table.set_index("varying factor")
        fig = go.Figure(
            data=go.Heatmap(
                z=df.values,
                x=[c.replace(" gap", "") for c in df.columns],
                y=df.index.values,
                text=df.values,
                texttemplate="%{text:.2f}",
                colorscale="blues",
                reversescale=False,
                textfont={"size": 24},
            )
        )
        fig.update_xaxes(title_text="Affected Factors")
        fig.update_yaxes(title_text="Varying Factors")
        eval_name = self.evaluation_type.replace("_", " ")
        fig.update_layout(
            title=f"{eval_name} slope changes",
            font=dict(
                size=22,
            ),
            height=800,
            width=1000,
        )
        return fig


class CompareSupervisionBoxPlot(PlotMixin):
    def __init__(self, runs: Runs):
        self.all_runs = runs

        self.compare_linear_ssl_table = tables.CompareSupervisionTable(
            runs, evaluation_type="linear_eval"
        ).table
        self.compare_linear_ssl_table["eval_type"] = "linear eval"

        self.compare_finetuning_ssl_table = tables.CompareSupervisionTable(
            runs, evaluation_type="finetuning"
        ).table
        self.compare_finetuning_ssl_table["eval_type"] = "finetuning"

        self.compare_ssl_table = self.compare_linear_ssl_table.append(
            self.compare_finetuning_ssl_table, ignore_index=True
        )
        self.compare_ssl_table["sup_eval_type"] = (
            self.compare_ssl_table["supervision"]
            + " ("
            + self.compare_ssl_table["eval_type"]
            + ")"
        )

    @property
    def title(self):
        # evaluation_type = self.evaluation_type.replace("_", " ").title()
        return f"Self-supervised versus Supervised Generalization Gaps"

    def plot(self) -> go.Figure:
        table = self.compare_ssl_table.melt("sup_eval_type")
        table = table.rename(
            columns={"variable": "Gap", "value": "Top-1 accuracy drop"}
        )
        table["Gap"] = table["Gap"].apply(lambda x: x.replace(" Gap", ""))
        box = px.box(
            table,
            x="Gap",
            y="Top-1 accuracy drop",
            color="sup_eval_type",
            color_discrete_map={  # replaces default color mapping by value
                "self-supervised (linear eval)": "#8ebcd1",
                "self-supervised (finetuning)": "#327fa8",
                "supervised (finetuning)": "#db6a44",
                "supervised (linear eval)": "#d19f8e",
            },
        )
        box = self.stylize_pot(box)
        box.update_layout(
            legend_title_text="",
            title=self.title,
        )
        return box

    def save(self, save_dir: str = "plots/") -> go.Figure:
        fig = self.plot()
        save_path = os.path.join(save_dir, f"ssl_gaps.pdf")
        # workaround to avoid mathjax warning
        pio.full_figure_for_development(fig, warn=False)
        pio.kaleido.scope.mathjax = None
        fig.update_layout(width=1200, height=350)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.95,
                xanchor="left",
                x=0.0,
            ),
            margin=dict(r=10, b=5),
        )
        fig.write_image(save_path, format="pdf")
        return fig


class Compare21kBoxPlot(PlotMixin):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        self.all_runs = runs
        self.evaluation_type = evaluation_type

        self.compare_21k_linear_table = tables.Compare21kTable(
            runs, evaluation_type="linear_eval"
        ).table
        self.compare_21k_linear_table["eval_type"] = "linear eval"

        self.compare_21k_finetuning_table = tables.Compare21kTable(
            runs, evaluation_type="finetuning"
        ).table
        self.compare_21k_finetuning_table["eval_type"] = "finetuning"

        self.compare_21k_table = self.compare_21k_linear_table.append(
            self.compare_21k_finetuning_table, ignore_index=True
        )
        self.compare_21k_table["training_data_eval_type"] = (
            self.compare_21k_table["training_data"]
            + " ("
            + self.compare_21k_table["eval_type"]
            + ")"
        )

    @property
    def title(self):
        return f"Effect of larger training data on generalization"

    def plot(self) -> go.Figure:
        table = self.compare_21k_table.melt("training_data_eval_type")
        table = table.rename(
            columns={"variable": "Gap", "value": "Top-1 accuracy drop"}
        )
        table["Gap"] = table["Gap"].apply(lambda x: x.replace(" Gap", ""))
        box = px.box(
            table,
            x="Gap",
            y="Top-1 accuracy drop",
            color="training_data_eval_type",
            color_discrete_map={  # replaces default color mapping by value
                "ImageNet (linear eval)": "#79a86f",
                "ImageNet (finetuning)": "#2d5425",
                "ImageNet-21k (linear eval)": "#d6b076",
                "ImageNet-21k (finetuning)": "#875d1e",
            },
        )
        box = self.stylize_pot(box)
        box.update_layout(
            legend_title_text="",
            title=self.title,
        )
        return box

    def save(self, save_dir: str = "plots/"):
        fig = self.plot()
        # workaround to avoid mathjax warning
        pio.full_figure_for_development(fig, warn=False)
        pio.kaleido.scope.mathjax = None
        save_path = os.path.join(save_dir, f"21k_gaps.pdf")
        fig.update_layout(width=1200, height=320)
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=0.95,
                xanchor="left",
                x=0.0,
            ),
            margin=dict(r=10, b=5),
        )
        fig.write_image(save_path, format="pdf")
        return fig


class PairedFactorPlot(PlotMixin):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        self.all_runs = runs
        self.evaluation_type = evaluation_type

        self.table = tables.PairedFactorTable(
            runs, evaluation_type=evaluation_type
        ).table
        self.models = self.table["model"].unique().tolist()
        self.factor_display_names = [
            self.all_runs.FACTOR_TO_DISPLAY_NAME[f] for f in self.all_runs.FACTORS
        ]

    def plot_single_factor(
        self,
        factors_to_vary: List[str],
        fig: go.Figure = None,
    ) -> go.Figure:
        """Bar plots measuring the gaps of varying paired factors.

        Args:
            factors_to_vary: [Translation, Scale]
        """
        if fig is None:
            fig = go.Figure()
        df = self.table[self.table["factors_to_vary"] == ", ".join(factors_to_vary)]
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
        factor_to_vary_display = " and ".join(
            [self.all_runs.FACTOR_TO_DISPLAY_NAME[f] for f in factors_to_vary]
        )
        fig.update_xaxes(
            title_text=f"top-1 accuracy drop when {factor_to_vary_display} vary <br> ({self.evaluation_type.replace('_', ' ')})"
        )
        fig.update_layout(barmode="group")
        return fig

    def plot(self, factors_to_vary=["Translation", "Rotation"]) -> go.Figure:
        fig = self.plot_single_factor(factors_to_vary)
        fig = self.stylize_pot(fig)
        return fig

    def save_appendix(self, save_dir: str = "plots/"):
        for factors_to_vary_str in self.table["factors_to_vary"].unique().tolist():
            factors_to_vary = factors_to_vary_str.split(", ")
            fig = self.plot(factors_to_vary=factors_to_vary)
            save_path = os.path.join(
                save_dir,
                f"paired_{factors_to_vary_str.replace(', ', '_')}_{self.evaluation_type}_gaps.pdf",
            )
            fig.write_image(save_path, format="pdf")


class PairedFactorRelativePlot(PairedFactorPlot):
    def __init__(self, runs: Runs, evaluation_type: str = "linear_eval"):
        super().__init__(runs, evaluation_type=evaluation_type)
        self.gap_names = [f"{f} gap" for f in self.factor_display_names]

        self.baseline_table = self.find_baseline_table()
        self.relative_table = self.compute_relative_table()

    def find_baseline_table(self) -> pd.DataFrame:
        return tables.CanonicalTable(
            self.all_runs, evaluation_type=self.evaluation_type
        ).table

    def compute_relative_table(self):
        table = self.table.set_index(["factors_to_vary", "model"])
        cols = table.select_dtypes(np.number).columns
        table[cols] *= 100
        baseline_table = self.baseline_table.set_index("model")

        relative_table = table
        relative_table = table.subtract(baseline_table)
        relative_table = relative_table.reset_index()
        return relative_table

    def plot_single_factor(
        self,
        factors_to_vary: List[str],
        fig: go.Figure = None,
    ) -> go.Figure:
        """Bar plots measuring the gaps of varying paired factors.

        Args:
            factors_to_vary: [Translation, Scale]
        """
        if fig is None:
            fig = go.Figure()
        df = self.relative_table[
            self.relative_table["factors_to_vary"] == ", ".join(factors_to_vary)
        ]
        gap_names = [f"{f} gap" for f in self.factor_display_names]

        for model in self.models:
            model_df = df[df["model"] == model]
            base_model = self.get_base_model(model)
            fig.add_trace(
                go.Bar(
                    x=model_df[gap_names].values[0],
                    y=gap_names,
                    name=model,
                    orientation="h",
                    marker_pattern_shape="/" if "21k" in model else "",
                    marker_color=self.BASE_MODEL_TO_COLOR[base_model],
                )
            )
        factor_to_vary_display = " and ".join(
            [self.all_runs.FACTOR_TO_DISPLAY_NAME[f] for f in factors_to_vary]
        )
        fig.update_xaxes(
            title_text=f"relative top-1 accuracy drop when ({factor_to_vary_display} vary - no factors vary) <br> ({self.evaluation_type.replace('_', ' ')})"
        )
        fig.update_layout(barmode="group")
        return fig

    def plot(self, factors_to_vary=["Translation", "Rotation"]) -> go.Figure:
        fig = self.plot_single_factor(factors_to_vary)
        fig = self.stylize_pot(fig)
        return fig

    def save_appendix(self, save_dir: str = "plots/"):
        for factors_to_vary_str in self.table["factors_to_vary"].unique().tolist():
            factors_to_vary = factors_to_vary_str.split(", ")
            fig = self.plot(factors_to_vary=factors_to_vary)
            save_path = os.path.join(
                save_dir,
                f"paired_{factors_to_vary_str.replace(', ', '_')}_{self.evaluation_type}_gaps_relative_to_no_diversity.pdf",
            )
            fig.write_image(save_path, format="pdf")


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


class AllFactorRelativetPlot(AllFactorPlot):
    def __init__(
        self,
        runs: Runs,
        evaluation_type: str = "linear_eval",
        relative_to: str = "no_diversity",
    ):
        super().__init__(runs, evaluation_type=evaluation_type)
        self.runs = runs
        self.relative_to = relative_to
        # baseline table is 0-100
        self.baseline_table = self.find_baseline_table()
        self.models = self.baseline_table.model.unique().tolist()
        self.models.remove("Average")
        # multiply table by 100 to match
        cols = self.table.select_dtypes(np.number).columns
        self.table[cols] *= 100
        self.relative_table = self.compute_relative_table()

    def compute_relative_table(self):
        table = self.table.copy()
        # table["model"] = table["model"].apply(
        #     lambda x: self.runs.MODEL_TO_DISPLAY_NAME[x]
        # )
        table = table.set_index("model")
        baseline_line_table = self.baseline_table.set_index("model")
        relative_table = table.subtract(baseline_line_table, fill_value=0)
        relative_table = relative_table.reset_index()
        return relative_table

    def find_baseline_table(self) -> pd.DataFrame:
        if self.relative_to == "no_diversity":
            return tables.CanonicalTable(
                self.all_runs, evaluation_type=self.evaluation_type
            ).table
        raise ValueError(f"relative to {self.relative_to} is not recognized")

    def plot(
        self,
    ) -> go.Figure:
        """Bar plots measuring the gaps of factors when all are seen with diversity"""
        fig = go.Figure()
        df = self.relative_table
        gap_names = [f"{f} gap" for f in self.factor_display_names]

        for model in self.models:
            model_df = df[df["model"] == model]
            base_model = self.get_base_model(model)

            fig.add_trace(
                go.Bar(
                    x=model_df[gap_names].values[0],
                    y=gap_names,
                    name=model,
                    orientation="h",
                    marker_pattern_shape="/" if "21k" in model else "",
                    marker_color=self.BASE_MODEL_TO_COLOR[base_model],
                )
            )
        fig.update_xaxes(
            title_text=f"Relative top-1 accuracy drop when (all factors vary - no factors vary) <br> ({self.evaluation_type.replace('_', ' ')})"
        )
        fig = self.stylize_pot(fig)
        return fig

    def save(self, save_dir: str = "plots/"):
        fig = self.plot()
        save_path = os.path.join(
            save_dir,
            f"all_factors_{self.evaluation_type}_gaps_relative_to_{self.relative_to}.pdf",
        )
        fig.write_image(save_path, format="pdf")


class AllFactorHeatmap(AllFactorPlot):
    def plot(
        self,
    ) -> go.Figure:
        """Bar plots measuring the gaps of factors when all are seen with diversity"""
        df = self.table
        gap_names = [f"{f} gap" for f in self.factor_display_names]

        z = df[gap_names].values.T
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=df["model"],
                y=gap_names,
                text=z,
                texttemplate="%{text:.2f}",
                colorscale="blues",
                reversescale=True,
            )
        )

        fig.update_xaxes(
            title_text=f"top-1 accuracy drop when all factors vary <br> ({self.evaluation_type.replace('_', ' ')})"
        )
        fig = self.stylize_pot(fig)
        return fig


class ClassSimilarityGapPlots(PlotMixin):
    def __init__(self):
        self.class_similarities = analyze_runs.ClassSimilarity()
        self.accuracies_df = self.class_similarities.accuracies_df
        self.accuracies_df["accuracy gap"] *= 100
        self.runs_cls = analyze_runs.Runs

    def plot_single_factor(self, factor: str) -> go.Figure:
        factor_display_name = self.runs_cls.FACTOR_TO_DISPLAY_NAME[factor]
        df = self.accuracies_df[
            self.accuracies_df["loader_name"] == f"val_diverse_{factor}"
        ]

        fig = px.scatter(
            df,
            x="similarity",
            y="accuracy gap",
            labels={"accuracy": f"{factor_display_name} accuracy"},
            hover_data=["class_name"],
            color="model",
            title=f"{factor_display_name} accuracy as class similarity changes",
        )
        fig = self.stylize_pot(fig)
        fig.update_layout(showlegend=False)
        return fig

    def plot_aggregate(self):
        df = self.accuracies_df

        fig = px.scatter(
            df,
            x="similarity",
            y="accuracy gap",
            # labels={"accuracy": f"accuracy"},
            hover_data=["class_name"],
            # color="model",
            title=f"generalization gaps as class similarity changes",
        )
        fig = self.stylize_pot(fig)
        fig.update_layout(showlegend=False)
        return fig

    def plot_aggregate_binned(self, bins: int = 5) -> go.Figure:
        x = self.accuracies_df.similarity.values
        y = self.accuracies_df["accuracy gap"].values
        means = binned_statistic(x, y, bins=bins).statistic
        sem_func = lambda x: stats.sem(x)
        sem = binned_statistic(x, y, statistic=sem_func, bins=bins).statistic

        fig = go.Figure(
            [
                go.Bar(
                    x=np.array(list(range(10))) / float(bins),
                    y=means,
                    error_y=dict(
                        type="data",  # value of error bar given in data coordinates
                        array=sem,
                        visible=True,
                    ),
                )
            ]
        )
        fig = self.stylize_pot(fig)
        fig.update_xaxes(title_text="Class similarity")
        fig.update_yaxes(title_text="Accuracy gap")
        return fig

    def plot(self):
        for factor in self.runs_cls.FACTORS:
            fig = self.plot_single_factor(factor)
            fig.show()

    def save(self, save_dir: str = "plots/"):
        for factor in self.runs_cls.FACTORS:
            fig = self.plot_single_factor(factor)

            save_path = os.path.join(
                save_dir,
                f"class_similarity_{factor}_gaps.pdf",
            )
            fig.write_image(save_path, format="pdf")

    def save_aggregate(self, save_dir: str = "plots/"):
        fig = self.plot_aggregate()
        fig.update_layout(width=1300, height=300)
        save_path = os.path.join(
            save_dir,
            f"class_similarity_gaps.pdf",
        )
        fig.write_image(save_path, format="pdf")

    def save_aggregate_binned(self, save_dir: str = "plots/"):
        fig = self.plot_aggregate_binned()
        fig.update_layout(width=1300, height=300)
        save_path = os.path.join(
            save_dir,
            f"class_similarity_gaps_binned.pdf",
        )
        fig.write_image(save_path, format="pdf")


if __name__ == "__main__":
    # run via: python analysis/plots.py
    dir_path = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(dir_path, "plots/")

    runs = Runs()

    # Hack to loadmathjax then overwirte later in the loop
    compare_ssl = CompareSupervisionBoxPlot(runs)
    compare_ssl.save(save_dir)

    for evaluation_type in ["linear_eval", "finetuning"]:
        # diverse_gaps = DataDiversityPlot(runs, evaluation_type=evaluation_type)
        # diverse_gaps.save(save_dir)
        # diverse_gaps.save_appendix(save_dir)
        # paired_gaps = PairedFactorPlot(runs, evaluation_type=evaluation_type)
        # paired_gaps.save_appendix(save_dir)
        # paired_relative_gaps = PairedFactorRelativePlot(
        #     runs, evaluation_type=evaluation_type
        # )
        # paired_relative_gaps.save_appendix(save_dir)

        # all_gaps = AllFactorPlot(runs, evaluation_type=evaluation_type)
        # all_gaps.save(save_dir)
        all_relative_gaps = AllFactorRelativetPlot(runs, evaluation_type=evaluation_type)
        all_relative_gaps.save(save_dir)

    compare_ssl = CompareSupervisionBoxPlot(runs)
    compare_ssl.save(save_dir)
    compare_21k = Compare21kBoxPlot(runs)
    compare_21k.save(save_dir)

    class_sim = ClassSimilarityGapPlots()
    class_sim.save_aggregate_binned(save_dir)
