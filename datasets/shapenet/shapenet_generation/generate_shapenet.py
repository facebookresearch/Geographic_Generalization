"""
Launch rendering code for a fixed number of objects from multiple categories 

Usage:
python multi_classes.py 
"""

import os
import argparse
import csv
import shutil
import json
import subprocess
from pathlib import Path
from tqdm import tqdm
import submitit
from typing import List, Set
from datasets.shapenet_generation import configs
from datetime import datetime


def run_locally(commands: List[str]):
    for command in tqdm(commands):
        run_command(command)


def run_command(command: str):
    result = subprocess.run(command.split(" "), capture_output=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        result.check_returncode()
        raise RuntimeError(f"rendering failed for command {command}")


def run_on_cluster_gpu(commands: List[str]):
    now = datetime.now()
    logs_dir = f"/checkpoint/{USER}/logs/shapenet-rendering/{now.strftime('%m-%d-%Y-%H-%M-%S')}"
    executor = submitit.SlurmExecutor(folder=logs_dir)
    executor.update_parameters(
        time=4300,
        gpus_per_node=1,
        cpus_per_task=10,
        partition="devlab",
    )
    job = executor.submit(run_locally, commands)
    print("Slurm job_id ", job.job_id)
    print("logs_dir ", logs_dir)
    output = job.result()


def run_on_cluster(commands: List[str]):
    """Run all commands in paralle using CPU"""
    now = datetime.now()
    logs_dir = f"/checkpoint/{USER}/logs/shapenet-rendering/{now.strftime('%m-%d-%Y-%H-%M-%S')}"
    executor = submitit.SlurmExecutor(folder=logs_dir)
    executor.update_parameters(
        time=4300,
        gpus_per_node=0,
        array_parallelism=512,
        cpus_per_task=2,
        partition="devlab",
    )
    jobs = executor.map_array(run_command, commands)
    print("Slurm job_id ", jobs[0].job_id)
    print("logs_dir ", logs_dir)
    outputs = [job.result() for job in jobs]


class Generation:
    def __init__(
        self,
        shapenet_dir: str = configs.DEFAULT_SHAPENET_DIR,
        synset_type: str = "overlapping",
        num_instances_per_synset: int = 50,
        num_views: int = 90,
        init_transform: str = "0 0 0",
        order: str = "XYZ",
        blender_command: str = configs.DEFAULT_BLENDER_COMMAND,
        out_dir: str = configs.DEFAULT_OUT_DIR,
    ):
        self.shapenet_dir = shapenet_dir
        self.synset_type = synset_type
        self.num_instances_per_synset = num_instances_per_synset
        self.num_views = num_views
        self.init_transform = init_transform
        self.order = order
        self.blender_command = blender_command
        self.out_dir = out_dir

        self.csv_file_path = os.path.join(self.out_dir, "fov.csv")
        self.ids_to_exclude: Set[str] = self.get_ids_to_exclude()
        self.synsets = self.get_synsets()

    def get_ids_to_exclude(self) -> Set[str]:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(dir_path, "duplicated_synsets.txt")) as f:
            ids_to_exclude = set(f.read().splitlines())
        return ids_to_exclude

    def get_synsets(self) -> List[str]:
        if self.synset_type == "all":
            list_dir = os.listdir(self.shapenet_dir)
            list_dir.remove("taxonomy.json")
            # remove synsets with all duplicates
            synsets = list(set(list_dir) - configs.SYNSETS_TO_EXCLUDE)
            return synsets
        elif self.synset_type == "overlapping":
            synsets = list(
                set(configs.IMAGENET_OVERLAPPING_SYNSETS) - configs.SYNSETS_TO_EXCLUDE
            )
            return synsets
        raise ValueError(f"{self.synset_type} not supported")

    def save_attributes(self):
        attributes_path = os.path.join(self.out_dir, "attributes.json")
        attributes = {
            "shapenet_dir": self.shapenet_dir,
            "synset_type": self.synset_type,
            "num_instances_per_synset": self.num_instances_per_synset,
            "num_views": self.num_views,
            "init_transform": self.init_transform,
            "order": self.order,
            "blender_c": self.blender_command,
            "out_dir": self.out_dir,
        }
        with open(attributes_path, "w") as f:
            json.dump(attributes, f)

    def create_fov_csv(self):
        header = [
            "class",
            "instance_id",
            "image_path",
            "pose_x",
            "pose_y",
            "pose_z",
        ]
        with open(self.csv_file_path, "w", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(header)  # write header

    def clean_up_directory(self):
        if Path(self.out_dir).exists():
            shutil.rmtree(self.out_dir)
        os.makedirs(self.out_dir, exist_ok=True)

    def get_instance_ids(self, synset: str) -> List[str]:
        """Excludes duplicated instances.

        Args:
            synset_dir: directory containing instances
        """
        synset_dir = os.path.join(self.shapenet_dir, synset)
        all_ids = set(os.listdir(synset_dir))
        filtered_ids = list(all_ids.difference(self.ids_to_exclude))

        if len(filtered_ids) < self.num_instances_per_synset:
            raise ValueError(
                f"synset {synset} has fewer than {self.num_instances_per_synset} instances"
            )
        return filtered_ids[: self.num_instances_per_synset]

    def build_commands(self) -> List[str]:
        commands = []

        for synset in self.synsets:
            synset_dir = os.path.join(self.shapenet_dir, synset)
            if not os.path.isdir(synset_dir):
                raise FileNotFoundError(f"synset dir {synset_dir} does not exist")

            synset_out_dir = os.path.join(self.out_dir, synset)
            global_command = f"{self.blender_command} --python-use-system-env --background -noaudio --python render.py -- --num_views {self.num_views} --out_dir {synset_out_dir} --init_transfo {self.init_transform} --order {self.order} --csv_file {self.csv_file_path}"
            obj_dirs = self.get_instance_ids(synset)

            for obj in obj_dirs:
                command = (
                    global_command
                    + f" --synset_id {synset} --obj_id {obj} --obj_path models/model_normalized.obj"
                )
                commands.append(command)
        return commands

    def run(self, on_cluster: bool = False):
        self.clean_up_directory()
        self.save_attributes()
        self.create_fov_csv()
        commands = self.build_commands()
        print(len(commands), "commands to run")

        if not on_cluster:
            run_locally(commands)
        else:
            run_on_cluster(commands)

        print("Done!")


def parse_args(parser):
    args = parser.parse_args()
    print(args)
    return args


if __name__ == "__main__":
    USER = os.getenv("USER")
    # generated via pd.read_csv(synset.txt, delimiter="\t", names=["synset_id", "human_names", "children"], dtype={"synset_id": str})

    parser = argparse.ArgumentParser(
        description="Launch rendering code for multiple classes"
    )
    parser.add_argument(
        "--synset_type",
        default="overlapping",
        choices=["all", "overlapping"],
        help="set of synsets to use",
    )
    parser.add_argument(
        "--shapenet_dir",
        type=str,
        default=configs.DEFAULT_SHAPENET_DIR,
        help="directory where ShapeNet obj are stored",
    )
    parser.add_argument(
        "--blender_command",
        type=str,
        default=configs.DEFAULT_BLENDER_COMMAND,
        help="command to launch blender",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=configs.DEFAULT_OUT_DIR,
        help="directory where pngs will be saved",
    )
    parser.add_argument(
        "--num_instances_per_synset",
        type=int,
        default=50,
        help="number of object per synset to use",
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=90,
        help="number of views to render",
    )
    parser.add_argument(
        "--init_transform",
        type=str,
        default="0 0 0",
        help="initial_transform that modifies canonical orientation",
    )
    parser.add_argument(
        "--order",
        type=str,
        default="XYZ",
        help="order of rotation",
    )
    parser.add_argument(
        "--cluster", default=False, action="store_true", help="run on cluster"
    )

    args = parse_args(parser)

    generation = Generation(
        synset_type=args.synset_type,
        shapenet_dir=args.shapenet_dir,
        blender_command=args.blender_command,
        out_dir=args.out_dir,
        num_instances_per_synset=args.num_instances_per_synset,
        num_views=args.num_views,
        init_transform=args.init_transform,
        order=args.order,
    )
    generation.run(on_cluster=args.cluster)
