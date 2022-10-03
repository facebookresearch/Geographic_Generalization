"""
Collect synsets that are both in ImageNet classes and ShapeNet categories
Output the list in a .txt file with 3 columns (synsetId, name, children if any) 

Usage:
python parse_classes.py
"""

import os
import json
import numpy as np

shapenet_directory = "/datasets01/ShapeNetCore.v2/080320/"
imagenet_directory = "/datasets01/imagenet_full_size/061417/"

with open(os.path.join(shapenet_directory, "taxonomy.json")) as f:
    shapenet_taxonomy = json.load(f)

with open(os.path.join(imagenet_directory, "labels.txt")) as f:
    imagenet_classes = np.loadtxt(f, delimiter=",", dtype=str)

shapenet_dict = {}
for i in shapenet_taxonomy:
    shapenet_dict[i["synsetId"]] = {}
    shapenet_dict[i["synsetId"]]["name"] = str(i["name"])
    shapenet_dict[i["synsetId"]]["children"] = i["children"]

classes = []
with open("synsets.txt", "w") as f:
    for c in os.listdir(shapenet_directory):
        if os.path.isdir(os.path.join(shapenet_directory, c)):
            name = shapenet_dict[c]["name"]
            children = shapenet_dict[c]["children"]
            if "n" + c in imagenet_classes[:, 0]:
                print(c, name, children)
                s = f"{c}\t{name}\t{children}\n"
                f.write(s)
        # The following piece of code shows that bathtub and tub are two different classes of imagenet (tub being associated with in the second case vat see https://gist.github.com/yrevar/942d3a0ac09ec9e5eb3a), yet tub is part of the synset of bathtub in ShapeNet
        # -> using names can create errors
        # subnames=name.split(',')
        # for n in subnames:
        #     if n in imagenet_classes[:,1]:
        #         if n == 'tub':
        #             import pdb; pdb.set_trace()
