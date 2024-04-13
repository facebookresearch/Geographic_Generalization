"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
import urllib.request
import tarfile
import os
import argparse
import subprocess

AVAILABLE_DATASETS = [
    "imagenet_v2",
    "imagenet_o",
    "imagenet_r",
    "imagenet_sketch",
    "imagenet_a",
    "objectnet",
    "dollarstreet",
    "geode",
]


def download_data(all=True, datasets=[]):
    if all:
        datasets = AVAILABLE_DATASETS

    print(
        "\nInitiating the download of the following datasets:\n\t-",
        "\n\t- ".join(datasets),
        "\n\nConfirm by typing 0, or type anything else to exit",
    )
    x = input()

    if x == "0":
        if "imagenet_v2" in datasets:
            print("Downloading ImageNet-V2")
            urllib.request.urlretrieve(
                "https://huggingface.co/datasets/vaishaal/ImageNetV2/resolve/main/imagenetv2-matched-frequency.tar.gz",
                filename="data/imagenet_v2.tar",
            )
            print("Extracting data...")
            tar = tarfile.open("data/imagenet_v2.tar")
            os.mkdir("data/imagenet-v2/")
            tar.extractall(path="data/imagenet-v2/")
            tar.close()
            os.rename(
                "data/imagenet-v2/imagenetv2-matched-frequency-format-val/",
                "data/imagenet-v2/test/",
            )

        elif "imagenet_r" in datasets:
            print("Downloading ImageNet-R")
            urllib.request.urlretrieve(
                "https://people.eecs.berkeley.edu/~hendrycks/imagenet-r.tar",
                filename="data/imagenet_r.tar",
            )
            print("Extracting data...")
            os.mkdir("data/imagenet_r/")
            tar = tarfile.open("data/imagenet_r.tar")
            tar.extractall(path="data/imagenet_r/")
            os.rename("data/imagenet_r/imagenet_r/", "data/imagenet_r/test/")
            tar.close()

        elif "imagenet_a" in datasets:
            print("Downloading ImageNet-A")
            urllib.request.urlretrieve(
                "https://people.eecs.berkeley.edu/~hendrycks/imagenet-a.tar",
                filename="data/imagenet_a.tar",
            )
            print("Extracting data...")
            os.mkdir("data/imagenet_a/")
            tar = tarfile.open("data/imagenet_a.tar")
            tar.extractall(path="data/imagenet_a")
            os.rename("data/imagenet_a/imagenet-a/", "data/imagenet_a/test/")
            tar.close()

        elif "imagenet_sketch" in datasets:
            print("Downloading ImageNet-Sketch")
            os.system(
                "gdown https://drive.google.com/u/0/uc?id=1Mj0i5HBthqH1p_yeXzsg22gZduvgoNeA"
            )
            print("Extracting data...")
            os.mkdir("data/imagenet_sketch/")
            os.system("mv ImageNet-Sketch.zip data/imagenet_sketch.zip")
            os.system("unzip data/imagenet_sketch.zip -d data/imagenet_sketch/")
            os.system("mv data/imagenet_sketch/sketch data/imagenet_sketch/test/")

        elif "objectnet" in datasets:
            print("Downloading ObjectNet")

            urllib.request.urlretrieve(
                "https://objectnet.dev/downloads/objectnet-1.0.zip",
                filename="data/objectnet.tar",
            )
            print("Extracting data...")
            os.mkdir("data/objectnet/")
            tar = tarfile.open("data/objectnet.tar")
            tar.extractall(path="data/objectnet/")
            tar.close()
            os.rename("data/objectnet/objectnet/", "data/objectnet/test/")

        elif "dollarstreet" in datasets:
            os.system("kaggle datasets download -d mlcommons/the-dollar-street-dataset")
            print("Extracting data...")
            os.system("mv the-dollar-street-dataset.zip data/dollarstreet.zip")
            os.system("unzip data/dollarstreet.zip -d data/dollarstreet/")

        elif "geode" in datasets:
            urllib.request.urlretrieve(
                "https://geodiverse-data-collection.cs.princeton.edu/geode.zip",
                filename="data/geode.zip",
            )
            print("Extracting data...")
            os.system("unzip data/geode.zip -d data/")
            os.system(
                "mv data/geode_metadata_test_1k_final.csv data/geode/metadata_test_1k_final.csv"
            )
        print("Datasets download complete, and can be found in the data folder")

    else:
        print("\nExiting the data download...")
        print(
            "\n** If you want to skip downloading data, or download only select benchmarks, open start.py and change the parameters under 'Step 1: Specify Dataset Downloads'.**\n"
        )
        exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-b",
        "--specific_benchmarks",
        nargs="+",
        help="Benchmarks To Download",
    )

    args = parser.parse_args()

    print("\nStarting your data set-up...\n")

    if args.specific_benchmarks:
        for benchmark in args.specific_benchmarks:
            if benchmark not in AVAILABLE_DATASETS:
                print(
                    f"ERROR: {benchmark} is not currently an available benchmark. Please select from the following benchmark options: \n\n{', '.join(AVAILABLE_DATASETS)}\n"
                )
                exit()
        download_data(all=False, datasets=args.specific_benchmarks)
    else:
        download_data(all=True)
