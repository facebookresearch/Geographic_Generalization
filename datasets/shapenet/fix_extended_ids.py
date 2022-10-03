from shapenet_extended import (
    ShapeNetExtendedDataModule,
    SingleFovHandler,
)
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

    canonical_fov_handler = SingleFovHandler({})

    for setting in ["individual"]:
        data_dir = f"/checkpoint/garridoq/datasets_fov/{setting}/"

        # ######## CODE TO GENERATE
        # data_module = ShapeNetExtendedDataModule(
        #     data_dir=data_dir,
        #     canonical_fov_handler=canonical_fov_handler,
        #     diverse_fov_handler=canonical_fov_handler,
        # )
        # split(data_module, "./")

    ######## CODE TO USE IT
    data_module = ShapeNetExtendedDataModule(
        data_dir=data_dir,
        canonical_fov_handler=canonical_fov_handler,
        diverse_fov_handler=canonical_fov_handler,
        trainval_ids_file="./trainval_ids.txt",
        test_ids_file="./test_ids.txt",
    )
    data_module.setup()
