import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
import os

font = {"weight": "normal", "size": 12}

# Make universal color and marker maps
model_names = ["resnet18", "resnet50", "resnet101", "mlpmixer", "vit"]

COLORDICT = {
    "dummy": "grey",
    "imagenet": "red",
    "imagenetv2": "brown",
    "objectnet": "green",
    "imageneta": "purple",
    "imagenetr": "pink",
    "imagenetsketch": "orange",
    "dollarstreet": "blue",
    "dollarstreet-high": "#0d7add",
    "dollarstreet-middle": "#55a1e7",
    "dollarstreet-low": "#b6d7f4",
    "dollarstreet-africa": "#21f0ea",
    "dollarstreet-europe": "#1ac0bb",
    "dollarstreet-the americas": "#17a8a3",
    "dollarstreet-asia": "#0d605d",
}

markers = ["*", "X", "D", "s", "v", "o"]

MARKERDICT = dict(zip(model_names, markers[0 : len(model_names)]))
plt.rc("font", **font)


def make_performance_comparison_plots_across_models(
    results, filter_str="_test_accuracy", anti_filter_str="-"
):
    generalization_cols = [
        x for x in results.columns if filter_str in x and anti_filter_str not in x
    ]
    generalization_names = [
        x.split(filter_str)[0] for x in list(results[generalization_cols].keys())
    ]

    x = np.arange(len(generalization_names))
    width = 0.1
    fig, ax = plt.subplots(figsize=(14, 8))

    for i in range(len(results)):
        row = results.iloc[i]
        generalization_vals = row[generalization_cols].tolist()
        bar_plt = plt.bar(
            x + i * width, generalization_vals, label=row["Model"], width=width
        )
        # plt.bar_label(bar_plt, padding=3, fmt="%.2f")

    if len(results) > 1:
        ax.set_xticks(ticks=x + 0.5 * (len(results) - 1) * width)
    else:
        ax.set_xticks(ticks=x)
    ax.set_xticklabels(generalization_names)

    plt.xlabel("Dataset")
    plt.ylabel("Top 1 Accuracy")
    plt.title("Model Performance Comparison Across Datasets")
    plt.legend(loc="upper right")
    plt.show()
    return fig


def make_property_vs_benefit_plot_across_models(
    results,
    property_name,
    filter_str,
    threshold=0,
    select_datasets=False,
    datasets_to_select=[],
    select_models=False,
    models_to_select=[],
):
    colordict = COLORDICT
    markerdict = MARKERDICT

    if select_datasets:
        cols = [
            x
            for x in results.columns
            if x.split("_")[0].split("-")[0] in datasets_to_select
        ]
        results = results[cols + ["Model"]]

    if select_models:
        results = results[results["Model"].isin(models_to_select)]
        markerdict = {k: MARKERDICT[k] for k in models_to_select}

    # fig = plt.figure(figsize=(14, 8))
    fig, ax = plt.subplots(figsize=(14, 8))
    property_cols = [x for x in results.columns.values if filter_str in x]

    for i in range(len(results)):
        row = results.iloc[i]
        model = row["Model"]

        # Get property data
        property_vals = row[property_cols].values.tolist()
        property_dict = {
            property_cols[i].split(filter_str)[0]: property_vals[i]
            for i in range(len(property_vals))
        }

        property_df = (
            pd.DataFrame.from_dict(property_dict, orient="index")
            .rename(columns={0: property_name.capitalize()})
            .reset_index()
        )

        # Get generalization data
        generalization_cols = [
            x for x in results.columns if "accuracy" in x and "dollarstreet-" not in x
        ]
        generalization_vals = row[generalization_cols].tolist()
        generalization_dict = {
            generalization_cols[i].split("_test_accuracy")[0]: generalization_vals[i]
            for i in range(len(generalization_cols))
        }

        generalization_df = (
            pd.DataFrame.from_dict(generalization_dict, orient="index")
            .rename(columns={0: "Accuracy"})
            .reset_index()
        )

        # Get detailed fairness data
        if select_datasets and datasets_to_select == ["dollarstreet"]:
            fairness_cols = [x for x in results.columns if "dollarstreet-" in x]
        else:
            fairness_cols = [
                x
                for x in results.columns
                if "dollarstreet-" in x and ("low" in x or "africa" in x)
            ]

        fairness_vals = row[fairness_cols].tolist()

        fairness_dict = {
            fairness_cols[i].replace("_test_accuracy", ""): fairness_vals[i]
            for i in range(len(fairness_cols))
        }
        fairness_df = (
            pd.DataFrame.from_dict(fairness_dict, orient="index")
            .rename(columns={0: "Accuracy"})
            .reset_index()
        )

        # Combine and make colors
        benefits_df = pd.concat([fairness_df, generalization_df])

        benefits_df["Base_Dataset"] = benefits_df["index"].apply(
            lambda x: x.split("_")[0].split("-")[0]
        )  # e.g 'Dollarstreet'
        benefits_df["Specific_Dataset"] = benefits_df["index"].apply(
            lambda x: x.split("_")[0]
        )  # 'Dollarstreet-africa'

        res = (
            pd.merge(
                property_df,
                benefits_df,
                how="left",
                left_on="index",
                right_on="Base_Dataset",
            )
            .dropna()
            .drop(columns=["index_y", "index_x"])
        )

        res["Color"] = res["Specific_Dataset"].apply(lambda x: colordict[x])

        # Plot
        plt.scatter(
            x=res[property_name.capitalize()],
            y=res["Accuracy"],
            c=res["Color"],
            marker=markerdict[model],
            s=200,
        )

    m_handles = [
        Line2D(
            [],
            [],
            marker=v,
            color="w",
            markerfacecolor="k",
            markeredgecolor="k",
            label=k,
            markersize=12,
        )
        for k, v in markerdict.items()
        if k in results["Model"].tolist()
    ]

    c_handles = [
        Line2D(
            [0.01],
            [0.01],
            marker="o",
            color="w",
            markerfacecolor=v,
            label=k,
            markersize=12,
        )
        for k, v in colordict.items()
        if k in res["Specific_Dataset"].tolist()
    ]
    color_legend = plt.legend(
        title="Dataset",
        handles=c_handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
    )
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax = plt.gca().add_artist(color_legend)
    plt.legend(
        title="Model", handles=m_handles, bbox_to_anchor=(1.05, 0.35), loc="upper left"
    )

    if threshold != None:
        plt.xlabel(f"{property_name.capitalize()} at Threshold = {threshold}")
        plt.title(f"{property_name.capitalize()} v.s Accuracy, (T={threshold})")
    else:
        plt.xlabel(f"{property_name.capitalize()}")
        plt.title(f"{property_name.capitalize()} v.s Accuracy")
    plt.ylabel("Test Accuracy")
    plt.show()
    return fig


def make_threshold_analysis_plots(results, property_name="sparsity"):
    cols = [x for x in results.columns.values if property_name in x or "Model" in x]
    subset = results[cols]

    plots = {}

    for index, row in results.iterrows():
        datasets_used = list(
            set([x.split("_")[0] for x in subset.columns.values if "Model" not in x])
        )

        property_values = []
        threshold_values = []
        dataset_values = []

        for dataset in datasets_used:
            dataset_df = row[[x for x in cols if dataset in x]]

            dataset_values = dataset_values + [
                x.split("_")[0].split("-")[0] for x in dataset_df.keys()
            ]
            threshold_values = threshold_values + [
                x.split("_")[-1] for x in dataset_df.keys()
            ]
            property_values = property_values + dataset_df.values.tolist()

        df = pd.DataFrame(
            {
                "Property": property_values,
                "Dataset": dataset_values,
                "Threshold": threshold_values,
            }
        )
        df["Color"] = df["Dataset"].apply(lambda x: COLORDICT[x])

        # Plot
        fig, ax = plt.subplots(figsize=(14, 8))
        plt.scatter(
            x=df["Threshold"],
            y=df["Property"],
            c=df["Color"],
            s=200,
        )

        c_handles = [
            Line2D(
                [0.01],
                [0.01],
                marker="o",
                color="w",
                markerfacecolor=v,
                label=k,
                markersize=12,
            )
            for k, v in COLORDICT.items()
            if k in df["Dataset"].tolist()
        ]
        color_legend = plt.legend(
            title="Dataset",
            handles=c_handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
        )
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax = plt.gca().add_artist(color_legend)
        plt.xlabel(f"Threshold")
        plt.title(
            f"{property_name.capitalize()} Values by Threshold Across Datasets, {row['Model'].capitalize()}"
        )
        plt.ylabel(f"{property_name.capitalize()} Value")
        plots[f"{property_name.capitalize()}_Thresholds_{row['Model']}"] = fig

    return plots


def save_plots(plots={}, log_dir=""):
    os.makedirs(os.path.join(log_dir, "plots"), exist_ok=True)
    plots_path = os.path.join(log_dir, "plots")
    for plot_name, image in plots.items():
        path = os.path.join(plots_path, plot_name)
        image.savefig(path)

    print(f"Plots Saved to {plots_path}")


def make_experiment_plots(results, log_dir=""):
    plots = {}

    # Make benefits graph
    included_benefits = [
        x
        for x in results.columns.values
        if ("test_accuracy" in x and "-" not in x) or "Model" in x
    ]
    if included_benefits != []:
        plots[
            "Performance_Comparison.JPEG"
        ] = make_performance_comparison_plots_across_models(
            results, filter_str="_test_accuracy", anti_filter_str="-"
        )

    # Make property-benefit graphs

    included_properties = list(
        set(
            [
                x.split("_")[1]
                for x in results.columns.values
                if "accuracy" not in x and "Model" not in x
            ]
        )
    )
    properties_with_thresholds = ["sparsity"]
    print(included_properties)

    if included_properties != [] and included_benefits != []:
        for property_name in included_properties:

            # Make property-benefit plots at different thresholds
            if property_name in properties_with_thresholds:
                thresholds = [
                    x.split(f"{property_name}_")[-1]
                    for x in results.columns.values
                    if f"{property_name}" in x
                ]
                thresholds = list(set(thresholds))

                for threshold in thresholds:
                    plots[
                        f"{property_name.capitalize()}_{threshold}.JPEG"
                    ] = make_property_vs_benefit_plot_across_models(
                        results=results,
                        property_name=property_name,
                        threshold=threshold,
                        filter_str=f"_{property_name}_{threshold}",
                        select_datasets=False,
                        select_models=False,
                    )
                    # If dollarstreet is included and we measured this property on dollarstreet, graph dollarstreet subsets in seperate graph
                    if any(
                        "dollarstreet" in string for string in results.columns.values
                    ) and any(
                        "dollarstreet_{property_name}" in string
                        for string in results.columns.values
                    ):
                        plots[
                            f"{property_name.capitalize()}_{threshold}_dollarstreet.JPEG"
                        ] = make_property_vs_benefit_plot_across_models(
                            results=results,
                            property_name=property_name,
                            threshold=threshold,
                            filter_str=f"_{property_name}_{threshold}",
                            select_datasets=True,
                            datasets_to_select=["dollarstreet"],
                            select_models=False,
                        )

            # Make property-benefit plot without thresholds
            else:
                plots[
                    f"{property_name.capitalize()}.JPEG"
                ] = make_property_vs_benefit_plot_across_models(
                    results=results,
                    property_name=property_name,
                    threshold=None,
                    filter_str=f"_{property_name}",
                    select_datasets=False,
                    select_models=False,
                )
                # If dollarstreet is included and we measured this property on dollarstreet, graph dollarstreet subsets in seperate graph
                if any(
                    "dollarstreet" in string for string in results.columns.values
                ) and any(
                    "dollarstreet_{property_name}" in string
                    for string in results.columns.values
                ):
                    plots[
                        f"{property_name.capitalize()}_dollarstreet.JPEG"
                    ] = make_property_vs_benefit_plot_across_models(
                        results=results,
                        property_name=property_name,
                        threshold=None,
                        filter_str=f"_{property_name}",
                        select_datasets=True,
                        datasets_to_select=["dollarstreet"],
                        select_models=False,
                    )

    # Make plots for thresholds
    for property_with_threshold in properties_with_thresholds:
        threshold_plots = make_threshold_analysis_plots(
            results, property_name=property_with_threshold
        )
        plots.update(threshold_plots)

    if log_dir:
        save_plots(plots=plots, log_dir=log_dir)
    return plots


def generate_aggregate_plots(
    experiment_name="cluster_run",
    base_logging_path="/checkpoint/meganrichards/logs/interplay_project/",
):
    # Aggregate model results into one results CSV
    model_names = [
        x
        for x in os.listdir(os.path.join(base_logging_path, experiment_name))
        if "plot" not in x
    ]

    measurement_results = []
    for model_name in model_names:
        measurements = pd.read_csv(
            f"{os.path.join(base_logging_path, experiment_name)}/{model_name}/{model_name}_all_measures/measurements.csv",
            index_col=0,
        )
        measurement_results.append(measurements)

    results = pd.concat(measurement_results)

    # Generate plots
    make_experiment_plots(
        results, log_dir=os.path.join(base_logging_path, experiment_name)
    )
    return
