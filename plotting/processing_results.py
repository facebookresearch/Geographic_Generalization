import os
import pandas as pd
import wandb
from datasets.imagenet_classes import IMAGENET_CLASSES


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
        if "Experiment" in results and results["Experiment"] in experiment_name:
            results["name"] = run.name
            results_df = pd.DataFrame(results, index=[1])
            results_list.append(results_df)

    combined = pd.concat(results_list)
    model_names = []

    for n in combined["name"]:
        if len(n.split(f"{experiment_name}_")) > 1:
            model_names.append(n.split(f"{experiment_name}_")[1].split("_")[0])
        else:
            model_names.append(n.split(f"{experiment_name}_")[0].split("_")[0])

    combined["Model"] = model_names

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
        gap_dict[f"dollarstreet-gap_country-{percentile}"] = [gap]
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

    all_dfs = []
    for m in os.listdir(base):
        model_dir = os.path.join(base, m)
        model_name = m
        if os.path.isdir(model_dir):
            try:
                num_folder = [
                    x
                    for x in os.listdir(model_dir)
                    if x
                    not in ["measurements.csv", "plots", ".submitit", "multirun.yaml"]
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
                    res_with_countries,
                    percentiles=[0.05, 0.1, 0.25],
                    model_name=model_name,
                )
                all_dfs.append(gap_df)
            except Exception as e:
                print(e)

    combined = pd.concat(all_dfs)
    return combined


import pandas as pd
import os


def calculate_geode_averages(df, model_name):
    region_acc = df.groupby("region")["accurate_top5"].mean().to_dict()

    gap_dict = {}

    for region_name, region_accuracy in region_acc.items():
        gap_dict[f"geode-{region_name.lower()}_test_accuracy"] = [region_accuracy]

    gap_dict[f"geode-gap_region"] = [region_acc["Europe"] - region_acc["Africa"]]
    gap_dict[f"geode-worst"] = [min(region_acc.values())]
    gap_dict[f"geode-best"] = [max(region_acc.values())]

    gap_dict["Model"] = [model_name]
    gap_df = pd.DataFrame.from_dict(gap_dict, orient="columns")
    return gap_df


def calculate_geode_region_acc_and_gaps(
    base="/checkpoint/meganrichards/logs/interplay_project/geode_eval_04-10/",
):
    metadata = pd.read_csv("/checkpoint/meganrichards/datasets/geode/metadata_1k.csv")

    all_dfs = []
    for m in os.listdir(base):
        model_dir = os.path.join(base, m)
        model_name = m
        if os.path.isdir(model_dir):
            try:
                num_folder = [
                    x
                    for x in os.listdir(model_dir)
                    if x
                    not in ["measurements.csv", "plots", ".submitit", "multirun.yaml"]
                ][0]
                res = pd.read_csv(
                    os.path.join(
                        model_dir,
                        num_folder,
                        "GeodePerformance",
                        "geode_predictions.csv",
                    ),
                    index_col=0,
                ).reset_index()
                res_with_regions = pd.merge(
                    metadata, res, how="right", left_on="Unnamed: 0", right_on="index"
                )

                gap_df = calculate_geode_averages(
                    res_with_regions,
                    model_name=model_name,
                )
                all_dfs.append(gap_df)

            except Exception as e:
                print(e)

    combined = pd.concat(all_dfs)
    # combined.to_csv("/checkpoint/meganrichards/logs/interplay_project/geode_eval_04-10/geode_eval_combined.csv")
    return combined


def recalculate_geode_accuracy_with_new_labels():
    preds = (
        pd.read_csv(
            "/checkpoint/meganrichards/logs/interplay_project/geode_eval_04-10/resnet101/18/GeodePerformance/geode_predictions.csv"
        )
        .drop(columns=["id"])
        .reset_index()
        .rename(columns={"index": "id"})
    )
    metadata = pd.read_csv(
        "/checkpoint/meganrichards/datasets/geode/metadata_1k_test_new_labels.csv"
    )
    metadata["id"] = metadata.index
    combined = pd.merge(preds, metadata, how="left", on="id")
    from ast import literal_eval

    combined["predictions"] = combined["predictions"].apply(literal_eval)
    combined["new_1k_index"] = combined["new_1k_index"].apply(literal_eval)

    def calculate_acc5(x):
        # return (x['object_index']) in x['predictions']
        acc5 = len(set(x["predictions"]) & set(x["new_1k_index"])) > 0
        return acc5

    def calculate_acc1(x):
        # return (x['object_index']) in x['predictions']
        acc5 = len(set([x["predictions"][0]]) & set(x["new_1k_index"])) > 0
        return acc5

    combined["accurate_top5"] = combined.apply(calculate_acc5, axis=1)
    combined["accurate_top1"] = combined.apply(calculate_acc1, axis=1)
    # #combined[~combined['object'].isin(['waste_container', 'stall', 'streetlight_lantern'])]
    return combined


def remapping_geode():
    GEODE_CLASSES_TO_IMAGENET_CLASSES = {
        "bag": ["backpack", "purse", "punching bag", "sleeping bag", "plastic bag"],
        "hand soap": ["soap dispenser", "lotion"],
        "dustbin": ["bucket"],
        "toothbrush": [],
        "toothpaste toothpowder": [],
        "hairbrush comb": ["hair clip"],
        "chair": ["barber chair", "folding chair", "rocking chair", "couch"],
        "hat": ["cowboy hat", "swimming cap", "football helmet"],
        "light fixture": ["table lamp"],
        "light switch": ["electrical switch"],
        "plate of food": ["meatloaf", "soup bowl", "plate"],
        "spices": [],
        "stove": ["Dutch oven", "stove"],
        "cooking pot": ["frying pan", "hot pot", "Crock Pot"],
        "cleaning equipment": ["vacuum cleaner", "washing machine"],
        "lighter": ["lighter"],
        "medicine": ["pill bottle", "medicine cabinet"],
        "candle": ["candle"],
        "toy": ["teddy bear"],
        "jug": ["water jug", "whiskey jug", "water bottle"],
        "streetlight lantern": ["lighthouse", "torch"],
        "front door": ["sliding door"],
        "tree": [],
        "house": ["cliff dwelling", "mobile home", "barn", "home theater"],
        "backyard": ["patio"],
        "truck": ["garbage truck", "semi-trailer truck", "tow truck"],
        "waste container": ["plastic bag"],
        "car": [
            "garbage truck",
            "recreational vehicle",
            "semi-trailer truck",
            "tow truck",
            "sports car",
            "railroad car",
        ],
        "fence": ["chain-link fence", "picket fence", "split-rail fence"],
        "road sign": ["traffic or street sign"],
        "dog": [
            "Bernese Mountain Dog",
            "Sealyham Terrier",
            "Toy Poodle",
            "toy terrier",
            "African wild dog",
            "husky",
            "Maltese",
            "Beagle",
            "Labrador Retriever",
            "Cairn Terrier",
        ],
        "wheelbarrow": ["wheelbarrow"],
        "religious building": ["mosque", "church"],
        "stall": ["toilet seat"],
        "boat": ["motorboat", "canoe"],
        "monument": ["triumphal arch", "obelisk", "stupa"],
        "flag": ["flagpole"],
        "bus": ["minibus", "school bus"],
        "storefront": ["traffic or street sign", "grocery store"],
        "bicycle": ["tricycle", "mountain bike"],
    }

    metadata = pd.read_csv(
        "/checkpoint/meganrichards/datasets/geode/metadata_1k_test.csv"
    )  # .reset_index().rename(columns = {'index': 'id'})
    metadata["id"] = metadata.index
    new_id_map = {}
    for k, v in GEODE_CLASSES_TO_IMAGENET_CLASSES.items():
        index_list = []
        for v_i in v:
            index_list.append(IMAGENET_CLASSES.index(v_i))
        new_id_map[k] = index_list

    def use_new_id_map(x):
        return new_id_map[x.replace("_", " ")]

    metadata["new_1k_index"] = metadata["object"].apply(use_new_id_map)
    return metadata


import os
import pandas as pd
from ast import literal_eval


def calculate_acc5(x):
    # return (x['object_index']) in x['predictions']
    acc5 = len(set(x["predictions"]) & set(x["new_1k_index"])) > 0
    return acc5


def calculate_acc1(x):
    # return (x['object_index']) in x['predictions']
    acc5 = len(set([x["predictions"][0]]) & set(x["new_1k_index"])) > 0
    return acc5


def recalculate_geode_group_accuracies(c, model_name=""):
    reg = c.groupby("region")["accurate_top1"].mean()

    gap_dict = {}
    gap_dict["Model"] = [model_name]
    for k, v in reg.items():
        gap_dict[f"geode-{k.lower()}_test_accuracy"] = [v]
    gap_dict["geode_test_accuracy"] = [c["accurate_top1"].mean()]
    gap_dict["geode-gap_europe_africa"] = [
        reg.to_dict()["Europe"] - reg.to_dict()["Africa"]
    ]
    gap_dict["geode-gap_americas_africa"] = [
        reg.to_dict()["Americas"] - reg.to_dict()["Africa"]
    ]
    gap_dict["geode-worst_region_test_accuracy"] = [reg.min()]
    gap_dict["geode-best_region_test_accuracy"] = [reg.max()]
    gap_dict["geode-gap_best_worst"] = [reg.max() - reg.min()]

    gap_df = pd.DataFrame.from_dict(gap_dict, orient="columns")

    return gap_df


def process_geode_results():
    base = "/checkpoint/meganrichards/logs/interplay_project/geode_eval_04-10/"
    metadata = pd.read_csv(
        "/checkpoint/meganrichards/datasets/geode/metadata_1k_test_new_labels.csv"
    )  # .reset_index().rename(columns = {'index': 'id'})
    metadata["id"] = metadata.index
    all_dfs = []
    for model in os.listdir(base):
        model_dir = os.path.join(base, model)
        if os.path.isdir(model_dir):
            try:
                num_folder = [
                    x
                    for x in os.listdir(model_dir)
                    if x
                    not in ["measurements.csv", "plots", ".submitit", "multirun.yaml"]
                ][0]
                preds = (
                    pd.read_csv(
                        os.path.join(
                            model_dir,
                            num_folder,
                            "GeodePerformance",
                            "geode_predictions.csv",
                        )
                    )
                    .drop(columns=["id"])
                    .reset_index()
                    .rename(columns={"index": "id"})
                )

                combined = pd.merge(preds, metadata, how="left", on="id")
                combined["predictions"] = combined["predictions"].apply(literal_eval)
                combined["new_1k_index"] = combined["new_1k_index"].apply(literal_eval)

                combined["accurate_top5"] = combined.apply(calculate_acc5, axis=1)
                combined["accurate_top1"] = combined.apply(calculate_acc1, axis=1)
                df = recalculate_geode_group_accuracies(combined, model)
                all_dfs.append(df)

            except Exception as e:
                print(e)

    l = pd.concat(
        all_dfs
    )  # .to_csv("/checkpoint/meganrichards/logs/interplay_project/geode_eval_04-10/geode_eval_combined.csv")
    return l


from plotting.plotting_utils import generate_generalizaton_fairness_comparison_combined
import pandas as pd


def make_geode_gap_plot():
    geode_gaps = pd.read_csv(
        "/checkpoint/meganrichards/logs/interplay_project/geode_eval_04-10/geode_eval_combined_corrected.csv",
        index_col=0,
    )
    rest = pd.read_csv(
        "/checkpoint/meganrichards/logs/interplay_project/more_models_04-03/combined_with_og_set.csv",
        index_col=0,
    )
    combined = pd.merge(geode_gaps, rest, how="inner", on="Model")
    combined.to_csv(
        "/checkpoint/meganrichards/logs/interplay_project/geode_eval_04-10/all_results_with_geode_corrected.csv"
    )
    combined = combined.rename(columns={"geode-gap_best_worst": "geode-gap_region"})
    generate_generalizaton_fairness_comparison_combined(
        disparity_type="region", fairness_dataset="geode", df=combined
    )
    return
