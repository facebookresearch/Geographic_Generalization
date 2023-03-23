import os
import pandas as pd
import wandb


def combine_model_results_from_file(
    base_dir="/checkpoint/meganrichards/logs/interplay_project/",
    folder_name="fairness_03-20",
):
    all_results = []
    results_folder = os.path.join(base_dir, folder_name)
    for model_name in os.listdir(results_folder):
        if ".csv" not in model_name:
            model_path = os.path.join(results_folder, model_name, "measurements.csv")
            model_results = pd.read_csv(model_path, index_col=0)
            all_results.append(model_results)

    all_results_df = pd.concat(all_results)
    all_results_df.to_csv(os.path.join(results_folder, "measurements.csv"))

    return all_results_df


def remove_prefix(x):
    if len(x.split("/")) > 1:
        return x.split("/")[-1]
    else:
        return x


def combine_model_results_from_wandb(
    experiment_name: str = "new_eval_fixes", save_path: str = ""
):
    api = wandb.Api()

    runs = api.runs("meganrichards/interplay_project")

    results_list = []
    for run in runs:

        results = run.summary._json_dict
        if "Experiment" in results and results["Experiment"].isin(experiment_name):
            results["name"] = run.name
            results_df = pd.DataFrame(results, index=[1])
            results_list.append(results_df)

    combined = pd.concat(results_list)
    combined["Model"] = combined["name"].apply(
        lambda x: x.split(f"{experiment_name}_")[1].split("_")[0]
    )
    combined = combined[
        [
            x
            for x in combined.columns.values
            if (x[0] != "_" and x != "Experiment" and x != "name")
        ]
    ]  # removing all _wandb, _timestamp, etc.
    filtered = combined.groupby("Model").aggregate(max)  # combine any duplicate runs
    filtered.groupby("Model").apply(lambda x: x.isna().sum().sum())
    filtered = filtered.reset_index()

    cols_map = {}
    for x in filtered.columns.values:
        cols_map[x] = remove_prefix(x)

    filtered.rename(columns=cols_map, inplace=True)

    if save_path:
        filtered.to_csv(save_path)

    return filtered


def calculate_gap(means, percentile):
    bottom_threshold = means.quantile(percentile)
    top_threshold = means.quantile(1 - percentile)
    bottom_vals = means[means < bottom_threshold]
    top_vals = means[means > top_threshold]
    assert abs(round(len(bottom_vals) / len(means))) <= 1.5
    gap = top_vals.mean() - bottom_vals.mean()
    assert gap > 0
    return gap, top_vals.mean(), bottom_vals.mean()


def calculate_percentile_gaps(results, percentiles, model_name):
    means = results.groupby("country.name")["accurate_top5"].mean()
    gap_dict = {}
    for percentile in percentiles:
        gap, top_avg, bottom_avg = calculate_gap(means, percentile)
        gap_dict[f"dollarstreet-gap-country-{percentile}"] = [gap]
        gap_dict[f"dollarstreet-country-top-{percentile}"] = [top_avg]
        gap_dict[f"dollarstreet-country-bottom-{percentile}"] = [bottom_avg]

    gap_dict["Model"] = [model_name]
    gap_df = pd.DataFrame.from_dict(gap_dict, orient="columns")
    return gap_df


def calculate_country_percentile_gaps(
    base="/checkpoint/meganrichards/logs/interplay_project/new_eval_fixes_03-12/",
):

    metadata = pd.read_csv(
        "/checkpoint/meganrichards/datasets/dollarstreet_kaggle/dataset_dollarstreet/images_v2_imagenet_test_with_income_and_region_groups.csv"
    )[["id", "country.name"]]
    res = pd.read_csv(
        "/checkpoint/meganrichards/logs/interplay_project/new_eval_fixes_03-12/resnet101/3/DollarStreetPerformance/dollarstreet_results.csv",
        index_col=0,
    )[["id", "accurate_top5"]]

    all_dfs = []
    for m in os.listdir(base):
        model_dir = os.path.join(base, m)
        model_name = m
        if os.path.isdir(model_dir):
            num_folder = [
                x
                for x in os.listdir(model_dir)
                if x not in ["measurements.csv", "plots", ".submitit", "multirun.yaml"]
            ][0]
            res = pd.read_csv(
                os.path.join(
                    model_dir,
                    num_folder,
                    "DollarStreetPerformance",
                    "dollarstreet_results.csv",
                ),
                index_col=0,
            )
            res_with_countries = pd.merge(metadata, res, how="right", on="id")
            gap_df = calculate_percentile_gaps(
                res_with_countries, percentiles=[0.05, 0.1, 0.25], model_name=model_name
            )
            all_dfs.append(gap_df)

    combined = pd.concat(all_dfs)
    return combined
