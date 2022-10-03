import os

USER = os.getenv("USER")


# note this excludes 02992529 (which is made of only duplicated instances)
# after exclusions below there are 15 overlapping synsets
IMAGENET_OVERLAPPING_SYNSETS = [
    "02747177",
    "02808440",
    "02843684",
    "03085013",
    "03207941",
    "03337140",
    "03642806",
    "03691459",
    "03710193",
    "03759954",
    "03761084",
    "03938244",
    "03991062",
    "04074963",
    "04090263",
    "04330267",
    "04554684",
]

synsets_with_all_duplicates = {"02992529"}
synsets_with_rendering_timeout = {"03337140", "04074963"}
SYNSETS_TO_EXCLUDE = synsets_with_all_duplicates.union(synsets_with_rendering_timeout)

DEFAULT_SHAPENET_DIR = "/datasets01/ShapeNetCore.v2/080320"
DEFAULT_BLENDER_COMMAND = f"/private/home/{USER}/blender-3.0.0-linux-x64/blender"
DEFAULT_OUT_DIR = f"/checkpoint/{USER}/datasets/shapenet_renderings"
