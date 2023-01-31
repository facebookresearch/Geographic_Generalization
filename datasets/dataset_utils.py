import os


def convert_imagenet_ids_to_class_folders(
    dataset_path="../../../../../checkpoint/meganrichards/datasets/imagenetv2-matched-frequency/val/",
    imagenet_labels_path="../../labels.txt",
):
    """Used to change folder names from class indexes (1,2,3) to imagenet class ids (N1283).

    Args:
        dataset_path (str): path from checkpoints directory where the dataset exists
        imagenet_labels_path: path to .txt file with imagenet labels following the structure of datasets01/imagenet_full_size/061417/labels.txt
    """
    # Change cwd to dataset location
    os.chdir(path=dataset_path)

    # Process imagenet labels into a list of strings with class folder names (['n123'])
    folder_to_str_dict = {}
    f = open(imagenet_labels_path)
    for line in f:
        key, value = line.split(",")[0:2]
        folder_to_str_dict[key] = value
    folder_list = list(folder_to_str_dict.keys())

    # Iterate dataset and change folder names
    for ind in os.listdir():
        try:
            if int(ind) % 100 == 0:
                print(ind)
            new_class = folder_list[int(ind)]
            os.rename(str(ind), new_class)
        except:
            pass
    print("Complete!")
