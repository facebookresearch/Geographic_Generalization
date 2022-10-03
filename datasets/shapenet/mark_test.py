from shapenet_extended import (
    SingleFovHandler,
    PairFovHandler,
    AllFovHandler,
    ShapeNetExtendedDataModule,
)
import numpy as np

setting = "individual"


fov_diverse = {
    "Translation": np.arange(0, 101, 1),
    "Rotation": np.arange(0, 101, 1),
    "Scale": np.arange(0, 101, 1),
    "Spot hue": np.arange(0, 101, 1),
    "Background path": [
        "/checkpoint/garridoq/datasets_fov/backgrounds/sky/0.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/sky/1.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/sky/2.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/sky/3.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/sky/4.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/home/0.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/home/1.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/home/2.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/home/3.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/home/4.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/grass/0.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/grass/1.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/grass/2.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/grass/3.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/grass/4.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/city/0.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/city/1.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/city/2.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/city/3.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/city/4.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/water/0.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/water/1.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/water/2.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/water/3.jpg",
        "/checkpoint/garridoq/datasets_fov/backgrounds/water/4.jpg",
    ],
}
diverse_fov_handler = SingleFovHandler(fov_diverse)

canonical_fov_handler = SingleFovHandler({})

datamodule = ShapeNetExtendedDataModule(
    data_dir=f"/checkpoint/garridoq/datasets_fov/{setting}",
    canonical_fov_handler=canonical_fov_handler,
    diverse_fov_handler=diverse_fov_handler,
    trainval_ids_file="./trainval_ids.txt",
    test_ids_file="./test_ids.txt",
)
