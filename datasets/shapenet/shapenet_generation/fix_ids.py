from datasets.shapenet import ShapeNetCanonicalDataModule, ShapeNetDataModule
import os
import pandas as pd


def split(data_module, save_folder=None):

    if save_folder is None:
        save_folder = data_module.data_dir

    synsets, instance_ids = data_module.get_instance_ids_and_synsets()
    synset_to_instance_ids = data_module.build_synset_to_instance_ids(
        synsets, instance_ids
    )

    # 80% training+validation, 20% for testing
    trainval_ids, test_ids = data_module._split_class_balanced(
        synset_to_instance_ids, prop=0.8
    )
    pd.DataFrame(trainval_ids).to_csv(
        os.path.join(save_folder, "trainval_ids.txt"),
        index=False,
        header=False,
        sep="\n",
    )
    pd.DataFrame(test_ids).to_csv(
        os.path.join(save_folder, "test_ids.txt"), index=False, header=False, sep="\n"
    )


if __name__ == "__main__":

    data_dir = (
        "/checkpoint/marksibrahim/datasets/shapenet_renderings_overlapping_small/"
    )

    ######## CODE TO GENERATE
    # data_module = ShapeNetDataModule(data_dir)
    # split(data_module, "/private/home/dianeb/video-variation/datasets/shapenet_generation/")

    ######## CODE TO USE IT
    data_module = ShapeNetDataModule(
        data_dir,
        trainval_ids_file="/private/home/dianeb/video-variation/datasets/shapenet_generation/trainval_ids.txt",
        test_ids_file="/private/home/dianeb/video-variation/datasets/shapenet_generation/test_ids.txt",
    )
    data_module.setup()

    data_module = ShapeNetCanonicalDataModule(
        data_dir,
        trainval_ids_file="/private/home/dianeb/video-variation/datasets/shapenet_generation/trainval_ids.txt",
        test_ids_file="/private/home/dianeb/video-variation/datasets/shapenet_generation/test_ids.txt",
    )
    data_module.setup()
