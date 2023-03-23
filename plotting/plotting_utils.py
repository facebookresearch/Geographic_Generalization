import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
import pandas as pd
import os
from scipy.stats import pearsonr

font = {"weight": "normal", "size": 12}

# Make universal color and marker maps
COLORDICT = {
    "dummy": "grey",
    "imagenet": "red",
    "imagenetv2": "brown",
    "dollarstreet": "blue",
    "dollarstreet-q1": "#0d7add",
    "dollarstreet-q2": "#55a1e7",
    "dollarstreet-q3": "#b6d7f4",
    "dollarstreet-q4": "#21f0ea",
    "dollarstreet-africa": "#21f0ea",
    "dollarstreet-europe": "#1ac0bb",
    "dollarstreet-the americas": "#17a8a3",
    "dollarstreet-asia": "#0d605d",
    "imagenetr": "pink",
    "imagenetsketch": "orange",
    "objectnet": "green",
    "imageneta": "purple",
}


MARKERDICT = {
    "resnet18": "v",
    "resnet34": "v",
    "resnet50": "^",
    "resnet101": "<",
    "resnet152": ">",
    "mlpmixer": "d",
    "vit": ".",
    "vitlarge": "o",
    "simclr": "D",
    "seer320": "p",
    "seer640": "h",
    "seer1280": "H",
    "clip-b16": "x",
    "clip-b32": "X",
    "convnext": "2",
}

plt.rc("font", **font)


def make_performance_comparison_plots_across_models(
    results, filter_str="_test_accuracy", anti_filter_str="-"
):

    results = results.sort_values(by="imagenet_test_accuracy", ascending=False)

    generalization_cols = [
        "imagenet_test_accuracy",
        "imagenetv2_test_accuracy",
        "imagenetr_test_accuracy",
        "dollarstreet_test_accuracy",
        "imagenetsketch_test_accuracy",
        "objectnet_test_accuracy",
        "imageneta_test_accuracy",
    ]
    generalization_names = [
        x.split(filter_str)[0] for x in list(results[generalization_cols].keys())
    ]

    x = np.arange(len(generalization_names))
    width = 0.04
    fig, ax = plt.subplots(figsize=(25, 8))

    for i in range(len(results)):
        row = results.iloc[i]
        generalization_vals = row[generalization_cols].tolist()
        bar_plt = plt.bar(
            x + i * width, generalization_vals, label=row["Model"], width=width
        )
        # plt.bar_label(bar_plt, padding=3, fmt="%.2f")

    if len(results) > 1:
        ax.set_xticks(ticks=x + 0.5 * (len(results)) * width)
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
    benefit_name,
    property_threshold=None,
    benefit_threshold=None,
    select_datasets=False,
    datasets_to_select=[],
    select_models=False,
    models_to_select=[],
    xlabel="",
    ylabel="",
    title="",
    xlim=(),
    ylim=(),
    calculate_model_corr=True,
    calculate_dataset_corr=True,
):
    # Define constants / assumptions
    colordict = COLORDICT
    markerdict = MARKERDICT

    easier_dataset_names = {
        "imagenet": "Val",
        "imageneta": "Adv",
        "imagenetr": "Rend",
        "imagenetv2": "V2",
        "imagenetsketch": "Sk",
        "objectnet": "Obj",
        "dollarstreet": "DS",
        "dollarstreet-q1": "DS-q1",
        "dollarstreet-q2": "DS-q2",
        "dollarstreet-q3": "DS-q3",
        "dollarstreet-q4": "DS-q4",
    }

    # Select results subset
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

    # Define filter strings to select relevant columns
    if property_threshold:
        property_filter_str = f"_{property_name}_{property_threshold}"
    else:
        property_filter_str = f"_{property_name}"

    if benefit_name.lower() == "fairness":
        results["dollarstreet_test_accuracy_gap"] = (
            results["dollarstreet-q1_test_accuracy"]
            - results["dollarstreet-q4_test_accuracy"]
        )
        benefit_filter_str = f"_test_accuracy_gap"
        benefit_name = "performance_gap_by_income"

    elif benefit_threshold:
        benefit_filter_str = f"_{benefit_name}_{benefit_threshold}"
    else:
        benefit_filter_str = f"_{benefit_name}"

    # Start a plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Define which columns to select, and analysis. Also which datasets are used (to be displayed)
    property_cols = [x for x in results.columns.values if property_filter_str in x]
    benefit_cols = [x for x in results.columns.values if benefit_filter_str in x]
    used_datasets = set([x.split("_")[0] for x in property_cols]) & set(
        [x.split("_")[0] for x in benefit_cols]
    )

    # Set up logging for correlations as they're calculated
    # bc we are iterating over models, we can compute directly in line and add to this dict
    model_corr = {}
    dataset_info_for_corr = {}  # we have to aggregate values instead, per dataset :(

    for dataset in list(easier_dataset_names.keys()):
        dataset_info_for_corr[dataset] = {"property": [], "benefit": []}

    # Iterate through each model's results
    for i in range(len(results)):
        row = results.iloc[i]
        model = row["Model"]
        print("Analyzing ", model)

        # Get property data
        property_vals = row[property_cols].values.tolist()
        property_dict = {
            property_cols[i].split(property_filter_str)[0]: property_vals[i]
            for i in range(len(property_vals))
        }

        property_df = (
            pd.DataFrame.from_dict(property_dict, orient="index")
            .rename(columns={0: property_name.capitalize()})
            .reset_index()
        )

        # Get benefits data
        benefits_vals = row[benefit_cols].tolist()
        benefits_dict = {
            benefit_cols[i].split(benefit_filter_str)[0]: benefits_vals[i]
            for i in range(len(benefit_cols))
        }

        benefits_df = (
            pd.DataFrame.from_dict(benefits_dict, orient="index")
            .rename(columns={0: benefit_name.capitalize()})
            .reset_index()
        )

        res = pd.merge(
            property_df,
            benefits_df,
            how="left",
            left_on="index",
            right_on="index",
        ).dropna()

        # Make dataset column for colors
        res["Base_Dataset"] = res["index"].apply(
            lambda x: x.split("_")[0].split("-")[0]
        )  # e.g 'Dollarstreet'
        res["Specific_Dataset"] = res["index"].apply(
            lambda x: x.split("_")[0]
        )  # 'Dollarstreet-africa'

        # Calculate / update correlation data
        if calculate_model_corr:
            corr, p = pearsonr(
                x=res[f"{property_name.capitalize()}"], y=res[benefit_name.capitalize()]
            )
            model_corr[model] = f"({corr:.2f}, {p:.2f})"

        for d in res["Specific_Dataset"].unique():
            dataset_info_for_corr[d]["property"].append(
                res[res["Specific_Dataset"] == d][property_name.capitalize()].item()
            )
            dataset_info_for_corr[d]["benefit"].append(
                res[res["Specific_Dataset"] == d][benefit_name.capitalize()].item()
            )

        res["Color"] = res["Specific_Dataset"].apply(lambda x: colordict[x])

        # Plot
        plt.scatter(
            x=res[property_name.capitalize()],
            y=res[benefit_name.capitalize()],
            c=res["Color"],
            marker=markerdict[model],
            s=200,
        )

    # Calculate data correlations
    if calculate_dataset_corr:
        dataset_corr = {}
        for dataset, values_dict in dataset_info_for_corr.items():
            if values_dict["property"]:
                corr, p = pearsonr(values_dict["property"], values_dict["benefit"])
                dataset_corr[dataset] = f"({corr:.2f}, {p:.2f})"

    m_handles = [
        Line2D(
            [],
            [],
            marker=v,
            color="w",
            markerfacecolor="k",
            markeredgecolor="k",
            label=f"{k} {model_corr[k] if calculate_model_corr else '   '}",
            markersize=12,
        )
        for k, v in markerdict.items()
        if k in results["Model"].tolist()
    ]

    for k in easier_dataset_names:
        if calculate_dataset_corr and not k in dataset_corr:
            dataset_corr[k] = "(na, na)"
    c_handles = [
        Line2D(
            [0.01],
            [0.01],
            marker="o",
            color="w",
            markerfacecolor=v,
            label=f"{easier_dataset_names[k]} {dataset_corr[k] if calculate_dataset_corr else '  '}",
            markersize=12,
        )
        for k, v in colordict.items()
        if k in used_datasets
    ]
    color_legend = plt.legend(
        title=f"{'Dataset (corr, p-val)' if calculate_dataset_corr else 'Dataset'}",
        handles=c_handles,
        bbox_to_anchor=(1.0, 1),
        loc="upper left",
        title_fontproperties={"size": 15},
        labelspacing=0.2,
    )
    color_legend._legend_box.align = "left"
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax = plt.gca().add_artist(color_legend)
    plt.legend(
        title=f"{'Model (corr, p-val)' if calculate_model_corr else 'Model      '}",
        handles=m_handles,
        bbox_to_anchor=(1.0, 0.6),
        loc="upper left",
        title_fontproperties={"size": 15},
        labelspacing=0.75,
    )
    property_pretty_name = " ".join([x.capitalize() for x in property_name.split("_")])
    benefit_pretty_name = " ".join([x.capitalize() for x in benefit_name.split("_")])
    # Set X label
    if xlabel:
        plt.xlabel(xlabel, fontdict={"size": 15})
    elif property_threshold != None:
        plt.xlabel(
            f"{property_name.capitalize()} at Threshold = {property_threshold}",
            fontdict={"size": 15},
        )
    else:
        plt.xlabel(f"{property_pretty_name}", fontdict={"size": 15})

    # Set Y label
    if ylabel:
        plt.ylabel(ylabel, fontdict={"size": 15})
    elif benefit_threshold != None:
        plt.xlabel(
            f"{benefit_name.capitalize()} at Threshold = {benefit_threshold}",
            fontdict={"size": 15},
        )
    else:
        plt.ylabel(benefit_pretty_name.capitalize(), fontdict={"size": 15})

    # Set Title

    if title:
        plt.title(f"{title}", fontdict={"size": 18})
    elif property_threshold != None or benefit_threshold != None:
        plt.title(
            f"{property_pretty_name} {f'(T={property_threshold})' if property_threshold else ''} vs {benefit_pretty_name} {f'T={benefit_threshold}' if benefit_threshold else ''}",
            fontdict={"size": 18},
        )
    else:
        plt.title(
            f"{property_pretty_name} v.s {benefit_pretty_name}",
            fontdict={"size": 18},
        )

    # Set X-Lim
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.tight_layout(pad=10)
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
        model = row["Model"]

        # Plot
        fig, ax = plt.subplots(figsize=(20, 10))
        plt.scatter(
            x=df["Threshold"],
            y=df["Property"],
            c=df["Color"],
            marker=MARKERDICT[model],
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
                        f"dollarstreet_{property_name}" in string
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
                print(
                    any("dollarstreet" in string for string in results.columns.values)
                    and any(
                        f"dollarstreet_{property_name}" in string
                        for string in results.columns.values
                    )
                )
                if any(
                    "dollarstreet" in string for string in results.columns.values
                ) and any(
                    f"dollarstreet_{property_name}" in string
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
        try:
            measurements = pd.read_csv(
                f"{os.path.join(base_logging_path, experiment_name)}/{model_name}/measurements.csv",
                index_col=0,
            )
            measurement_results.append(measurements)
        except Exception as e:
            print(str(e))

    results = pd.concat(measurement_results)

    # Generate plots
    make_experiment_plots(
        results, log_dir=os.path.join(base_logging_path, experiment_name)
    )
    return


from scipy import stats
import os


def calculate_corr_and_r2(
    results,
    property_name,
    benefit_name,
    property_threshold=None,
    benefit_threshold=None,
    select_datasets=False,
    datasets_to_select=[],
    select_models=False,
    models_to_select=[],
    calculate_model_corr=True,
    calculate_dataset_corr=True,
    save_dir="",
    verbose=False,
):
    # Define constants / assumptions
    easier_dataset_names = {
        "imagenet": "Val",
        "imageneta": "Adv",
        "imagenetr": "Rend",
        "imagenetv2": "V2",
        "imagenetsketch": "Sk",
        "objectnet": "Obj",
        "dollarstreet": "DS",
        "dollarstreet-q1": "DS-q1",
        "dollarstreet-q2": "DS-q2",
        "dollarstreet-q3": "DS-q3",
        "dollarstreet-q4": "DS-q4",
    }

    # Select results subset
    if select_datasets:
        cols = [
            x
            for x in results.columns
            if x.split("_")[0].split("-")[0] in datasets_to_select
        ]
        results = results[cols + ["Model"]]

    if select_models:
        results = results[results["Model"].isin(models_to_select)]

    # Define filter strings to select relevant columns
    if property_threshold:
        property_filter_str = f"_{property_name}_{property_threshold}"
    else:
        property_filter_str = f"_{property_name}"

    if benefit_name.lower() == "fairness":
        results["dollarstreet_test_accuracy_gap"] = (
            results["dollarstreet-q1_test_accuracy"]
            - results["dollarstreet-q4_test_accuracy"]
        )
        benefit_filter_str = f"_test_accuracy_gap"
        benefit_name = "performance_gap_by_income"

    elif benefit_threshold:
        benefit_filter_str = f"_{benefit_name}_{benefit_threshold}"
    else:
        benefit_filter_str = f"_{benefit_name}"

    if benefit_name.lower() == "test_accuracy":
        benefit_filter_str = f"_test_accuracy"
        benefit_name = "generalization"

    # Define which columns to select, and analysis. Also which datasets are used (to be displayed)
    property_cols = [x for x in results.columns.values if property_filter_str in x]
    benefit_cols = [x for x in results.columns.values if benefit_filter_str in x]
    if verbose:
        print(f"Property columns: {property_cols}")
        print(f"Benefit columns: {benefit_cols}")

    # Set up logging for correlations as they're calculated
    # bc we are iterating over models, we can compute directly in line and add to this dict
    model_corr = {}
    dataset_info_for_corr = {}  # we have to aggregate values instead, per dataset :(
    model_lin = {}

    for dataset in list(easier_dataset_names.keys()):
        dataset_info_for_corr[dataset] = {"property": [], "benefit": []}

    # Iterate through each model's results
    for i in range(len(results)):
        row = results.iloc[i]
        model = row["Model"]

        # Get property data
        property_vals = row[property_cols].values.tolist()
        property_dict = {
            property_cols[i].split(property_filter_str)[0]: property_vals[i]
            for i in range(len(property_vals))
        }

        property_df = (
            pd.DataFrame.from_dict(property_dict, orient="index")
            .rename(columns={0: property_name.capitalize()})
            .reset_index()
        )

        # Get benefits data
        benefits_vals = row[benefit_cols].tolist()
        benefits_dict = {
            benefit_cols[i].split(benefit_filter_str)[0]: benefits_vals[i]
            for i in range(len(benefit_cols))
        }
        benefits_df = (
            pd.DataFrame.from_dict(benefits_dict, orient="index")
            .rename(columns={0: benefit_name.capitalize()})
            .reset_index()
        )

        res = pd.merge(
            property_df,
            benefits_df,
            how="left",
            left_on="index",
            right_on="index",
        ).dropna()

        if verbose:
            print(res)

        # Make dataset column
        res["Base_Dataset"] = res["index"].apply(
            lambda x: x.split("_")[0].split("-")[0]
        )  # e.g 'Dollarstreet'
        res["Specific_Dataset"] = res["index"].apply(
            lambda x: x.split("_")[0]
        )  # 'Dollarstreet-africa'

        # Calculate / update correlation data
        if calculate_model_corr:
            x = res[f"{property_name.capitalize()}"]
            y = res[benefit_name.capitalize()]
            corr, p = stats.pearsonr(x=x, y=y)
            model_corr[model] = [corr, p]
            reg = stats.linregress(x, y)
            model_lin[model] = [reg.slope, reg.rvalue**2]

        for d in res["Specific_Dataset"].unique():
            dataset_info_for_corr[d]["property"].append(
                res[res["Specific_Dataset"] == d][property_name.capitalize()].item()
            )
            dataset_info_for_corr[d]["benefit"].append(
                res[res["Specific_Dataset"] == d][benefit_name.capitalize()].item()
            )

    # Calculate data correlations
    if calculate_dataset_corr:
        dataset_corr = {}
        dataset_lin = {}
        for dataset, values_dict in dataset_info_for_corr.items():
            if values_dict["property"]:
                x = values_dict["property"]
                y = values_dict["benefit"]
                corr, p = stats.pearsonr(x, y)
                dataset_corr[dataset] = [corr, p]
                reg = stats.linregress(x, y)
                dataset_lin[dataset] = [reg.slope, reg.rvalue**2]

    # Combine into model and dataset dataframes
    model_corr_df = pd.DataFrame.from_dict(
        model_corr, orient="index", columns=["Pearson_Correlation", "P_Value"]
    ).reset_index()
    model_lin_df = pd.DataFrame.from_dict(
        model_lin, orient="index", columns=["Slope", "R2"]
    ).reset_index()
    model_df = pd.merge(model_corr_df, model_lin_df, on="index", how="outer").rename(
        columns={"index": "Model"}
    )

    dataset_corr_df = pd.DataFrame.from_dict(
        dataset_corr, orient="index", columns=["Pearson_Correlation", "P_Value"]
    ).reset_index()
    dataset_lin_df = pd.DataFrame.from_dict(
        dataset_lin, orient="index", columns=["Slope", "R2"]
    ).reset_index()
    dataset_df = pd.merge(
        dataset_corr_df, dataset_lin_df, on="index", how="outer"
    ).rename(columns={"index": "Dataset"})

    if save_dir:
        if benefit_name == "performance_gap_by_income":
            if len(model_df) > 0:
                model_df.to_csv(
                    os.path.join(save_dir, f"{property_name}_fairness_models.csv")
                )
            if len(dataset_df) > 0:
                dataset_df.to_csv(
                    os.path.join(save_dir, f"{property_name}_fairness_datasets.csv")
                )
        else:
            if len(model_df) > 0:
                model_df.to_csv(
                    os.path.join(save_dir, f"{property_name}_{benefit_name}_models.csv")
                )
            if len(dataset_df) > 0:
                dataset_df.to_csv(
                    os.path.join(
                        save_dir, f"{property_name}_{benefit_name}_datasets.csv"
                    )
                )

    return model_df, dataset_df


def generate_generalizaton_fairness_comparison_combined(
    type: str = "Both",
    disparity_type: str = "income",
    country_threshold: float = 0.25,
    exclude_clip_and_seer=False,
    only_clip_and_seer=False,
    correlation_type="pearson",
    normalized=False,
):
    if exclude_clip_and_seer and only_clip_and_seer:
        raise Exception(
            "both parameters exclude_clip_and_seer and only_clip_and_seer are set to True, which is constradictory. Please set one of these to False."
        )
    if disparity_type not in ["income", "region", "country"]:
        raise Exception(
            "disparity_type parameter only includes the options: 'income', 'region', and 'country'"
        )
    if correlation_type not in ["pearson", "spearman"]:
        raise Exception(
            "correlation_type parameter only includes the options: 'pearson', 'spearman'"
        )

    COLORDICT = {
        "imagenet": "red",
        "imagenetv2": "brown",
        "dollarstreet": "blue",
        "dollarstreet-q1": "#0d7add",
        "dollarstreet-q2": "#55a1e7",
        "dollarstreet-q3": "#b6d7f4",
        "dollarstreet-q4": "#21f0ea",
        "dollarstreet-africa": "#21f0ea",
        "dollarstreet-europe": "#1ac0bb",
        "dollarstreet-the americas": "#17a8a3",
        "dollarstreet-asia": "#0d605d",
        "imagenetr": "pink",
        "imagenetsketch": "orange",
        "objectnet": "green",
        "imageneta": "purple",
    }
    filtered = pd.read_csv(
        "/checkpoint/meganrichards/logs/interplay_project/fairness_03-20/measurements_with_fairness_and_gaps_and_percentiles.csv",
        index_col=0,
    )
    filtered = filtered[~filtered["Model"].isin(["beit-base", "beit-large", "clip"])]
    if exclude_clip_and_seer:
        models_to_exclude = ["seer320", "seer640", "seer1280", "clip-b16", "clip-b32"]
        filtered = filtered[~filtered["Model"].isin(models_to_exclude)]
    elif only_clip_and_seer:
        models_to_include = ["seer320", "seer640", "seer1280", "clip-b16", "clip-b32"]
        filtered = filtered[filtered["Model"].isin(models_to_include)]

    if disparity_type == "income":
        benefit2_col = "dollarstreet-gap_income"
        benefit2_plot_name = "Income Disparity"
        normalizing_col = "dollarstreet-q1_test_accuracy"

    elif disparity_type == "region":
        benefit2_col = "dollarstreet-gap_region"
        benefit2_plot_name = "Region Disparity"
        normalizing_col = "dollarstreet-europe_test_accuracy"

    elif disparity_type == "country":
        benefit2_col = f"dollarstreet-gap_country-{country_threshold}"
        benefit2_plot_name = (
            f"Country Disparity (percentile = {int(100*country_threshold)}%)"
        )
        normalizing_col = f"dollarstreet-country-top-{country_threshold}"

    if type == "OOD":
        acc_cols = [
            "objectnet_test_accuracy",
            "imagenetr_test_accuracy",
            "imagenetsketch_test_accuracy",
            "imageneta_test_accuracy",
            "dollarstreet_test_accuracy",
        ]
    elif type == "ID":
        acc_cols = ["imagenet_test_accuracy", "imagenetv2_test_accuracy"]
    else:
        acc_cols = [
            "imagenet_test_accuracy",
            "imagenetv2_test_accuracy",
            "objectnet_test_accuracy",
            "imagenetr_test_accuracy",
            "imagenetsketch_test_accuracy",
            "imageneta_test_accuracy",
            "dollarstreet_test_accuracy",
        ]

    if normalized:
        cols_to_use = acc_cols + [benefit2_col] + [normalizing_col]
    else:
        cols_to_use = acc_cols + [benefit2_col]

    filtered = filtered[cols_to_use]

    fig, ax = plt.subplots(figsize=(14, 8))
    for benefit1_col in filtered.columns.values:
        if benefit1_col != benefit2_col and benefit1_col != normalizing_col:
            x = filtered[benefit1_col]
            y = filtered[benefit2_col]
            if normalized:
                y = y / filtered[normalizing_col]

            if correlation_type == "pearson":
                corr, p = stats.pearsonr(x, y)
            elif correlation_type == "spearman":
                corr, p = stats.spearmanr(x, y)
            corr_str = f"cor={corr:.2f}, (p={p:.3f})"

            dataset_name = benefit1_col.split("_test_accuracy")[0].capitalize()

            reg = stats.linregress(x, y)

            plt.scatter(x=x, y=y, color=COLORDICT[dataset_name.lower()])
            plt.axline(
                xy1=(0, reg.intercept),
                slope=reg.slope,
                color=COLORDICT[dataset_name.lower()],
                label=f"{dataset_name} | m={reg.slope:.2f}, (r2={reg.rvalue**2:.2f}) | {corr_str}",
            )

    easier_dataset_names = {
        "imagenet": "Val",
        "imageneta": "Adv",
        "imagenetr": "Rend",
        "imagenetv2": "V2",
        "imagenetsketch": "Sk",
        "objectnet": "Obj",
        "dollarstreet": "DS",
        "dollarstreet-q1": "DS-q1",
        "dollarstreet-q2": "DS-q2",
        "dollarstreet-q3": "DS-q3",
        "dollarstreet-q4": "DS-q4",
    }

    plt.xlabel(f"{'OOD Accuracy' if type == 'OOD'else 'ID Accuracy'}")
    plt.ylabel(benefit2_plot_name)
    plt.legend(title=f"Dataset, {correlation_type.capitalize()} Correlation")
    exclude_str = " - Excluding CLIP/SEER" if exclude_clip_and_seer else ""
    include_str = " - Only CLIP/SEER" if only_clip_and_seer else ""

    plt.title(
        f"{'OOD Accuracy' if type == 'OOD'else 'ID Accuracy'} v.s. {'Normalized' if normalized else ''} {benefit2_plot_name}{exclude_str}{include_str} {'- All Models' if not (exclude_str or include_str) else ''}"
    )
    plt.show()


def calculate_geography_income_gap_comparsion(geography_type="country"):
    filtered = pd.read_csv(
        "/checkpoint/meganrichards/logs/interplay_project/fairness_03-20/measurements_with_fairness_and_gaps.csv",
        index_col=0,
    )

    region = filtered["dollarstreet-gap_region"]
    country = filtered["dollarstreet-gap_country-0.25"]
    income = filtered["dollarstreet-gap_income"]

    fig, ax = plt.subplots(figsize=(14, 8))
    if geography_type == "region":
        x = region
    else:
        x = country

    y = income
    plt.scatter(x=x, y=y)
    reg = stats.linregress(x, y)
    corr, p = stats.pearsonr(x, y)
    corr_str = f"cor={corr:.2f}, (p={p:.3f})"

    plt.axline(
        xy1=(0, reg.intercept),
        slope=reg.slope,
        label=f"m={reg.slope:.2f}, (r2={reg.rvalue**2:.2f}) | {corr_str}",
    )
    plt.xlim(min(x) - 0.05, max(x) + 0.05)
    plt.ylim(min(y) - 0.05, max(y) + 0.05)
    plt.xlabel(f"Performance Gap By {geography_type.capitalize()}")
    plt.ylabel("Performance Gap by Income Quartile")
    plt.title(f"Performance Gap Comparison - {geography_type.capitalize()} vs Income")
    plt.legend()
    plt.show()


def make_task_improvement_plots(type="income", compare_gap_normalizing=False):
    filtered = pd.read_csv(
        "/checkpoint/meganrichards/logs/interplay_project/fairness_03-20/measurements_with_fairness_and_gaps_and_percentiles.csv",
        index_col=0,
    )
    filtered = filtered[~filtered["Model"].isin(["beit-base", "beit-large", "clip"])]

    for dataset_name in [
        "imagenet",
        "imagenetv2",
        "dollarstreet",
        "imageneta",
        "imagenetr",
        "imagenetsketch",
        "objectnet",
    ]:
        x = filtered[f"{dataset_name}_test_accuracy"]
        if type == "income":
            y1 = filtered["dollarstreet-q1_test_accuracy"]
            y2 = filtered["dollarstreet-q4_test_accuracy"]
        else:
            y1 = filtered["dollarstreet-europe_test_accuracy"]
            y2 = filtered["dollarstreet-africa_test_accuracy"]

        if compare_gap_normalizing:
            gap = y1 - y2
            gap_normalized = (y1 - y2) / y1
            y1 = gap
            y2 = gap_normalized

        corr1, p1 = stats.pearsonr(x, y1)
        corr1_str = f"cor={corr1:.2f}, (p={p1:.3f})"

        corr2, p2 = stats.pearsonr(x, y2)
        corr2_str = f"cor={corr2:.2f}, (p={p2:.3f})"

        reg1 = stats.linregress(x, y1)
        reg2 = stats.linregress(x, y2)

        if type == "income":
            line1_label = "Q1"
            line2_label = "Q4"
        elif type == "region":
            line1_label = "Europe"
            line2_label = "Africa"
        if compare_gap_normalizing:
            line1_label = "Gap            "
            line2_label = "Gap - Norm"

        plt.scatter(x=x, y=y1, color="blue")
        plt.axline(
            xy1=(0, reg1.intercept),
            slope=reg1.slope,
            color="blue",
            label=f"{line1_label}| m={reg1.slope:.2f}, (r2={reg1.rvalue**2:.2f}) | {corr1_str}",
        )
        plt.scatter(x=x, y=y2, color="orange")
        plt.axline(
            xy1=(0, reg2.intercept),
            slope=reg2.slope,
            color="orange",
            label=f"{line2_label}| m={reg2.slope:.2f}, (r2={reg2.rvalue**2:.2f}) | {corr2_str}",
        )
        plt.legend(loc="lower right")
        plt.ylim(0, 1)
        if compare_gap_normalizing:
            plt.title(
                f"{dataset_name.capitalize()} Acc vs DollarStreet Gaps (Regular and Normalized)"
            )
        else:
            plt.title(
                f"{dataset_name.capitalize()} Acc vs DollarStreet {type.capitalize()} Subset Acc"
            )
        plt.show()
