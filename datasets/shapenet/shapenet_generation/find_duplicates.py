"""
Collects the duplicated instances shared by multiple synsets in ShapeNet

Usage:
python find_duplicates.py
"""

import os
import json
import numpy as np

shapenet_directory = "/datasets01/ShapeNetCore.v2/080320/"
SAVE = True

with open(os.path.join(shapenet_directory, "taxonomy.json")) as f:
    shapenet_taxonomy = json.load(f)

shapenet_dict = {}
for i in shapenet_taxonomy:
    shapenet_dict[i["synsetId"]] = {}
    shapenet_dict[i["synsetId"]]["name"] = str(i["name"])
    shapenet_dict[i["synsetId"]]["children"] = i["children"]

shapenet_depth1_dict = {}
for synset in os.listdir(shapenet_directory):
    synset_folder = f"{shapenet_directory}/{synset}/"
    if os.path.isdir(synset_folder):
        shapenet_depth1_dict[synset] = {}
        shapenet_depth1_dict[synset]["instances"] = []
        for instance in os.listdir(synset_folder):
            shapenet_depth1_dict[synset]["instances"].append(instance)

seen = {}
repeated = {}
for synset in shapenet_depth1_dict.keys():
    for instance in shapenet_depth1_dict[synset]["instances"]:
        if instance in seen:
            if instance not in repeated.keys():
                # get the synset with which it was seen
                repeated[instance] = seen[instance]
            # add the synset with which it was newly seen
            repeated[instance].append(synset)
        else:
            seen[instance] = [synset]

print(
    "Over {0} instances, {1} are duplicated".format(
        len(seen.keys()), len(repeated.keys())
    )
)


if SAVE:
    with open("duplicated_synsets.txt", "w") as f:
        for item in repeated.keys():
            f.write(f"{item}\n")


print("For example,")
seen_subsets_with_duplicates = []
count = 0
for duplicate in repeated.keys():
    s = f"{duplicate} is seen in "
    if len(np.intersect1d(seen_subsets_with_duplicates, repeated[duplicate])) == 0:
        for j, synset in enumerate(repeated[duplicate]):
            if j == 0:
                s += synset
                s += " ("
                s += ",".join(
                    shapenet_dict[synset]["name"].split(",")[:2]
                )  # print only the first 2
                s += ")"
            else:
                s += " AND "
                s += synset
                s += " ("
                s += ",".join(
                    shapenet_dict[synset]["name"].split(",")[:2]
                )  # print only the first 2
                s += ")"
        print(s)
        count += 1
        seen_subsets_with_duplicates += repeated[duplicate]
    if count > 20:
        break
