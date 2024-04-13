"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""
from torchvision.datasets import ImageFolder
from torchvision.datasets.vision import VisionDataset
import os
import json
from PIL import Image
from datasets.image_datamodule import ImageDataModule


class ObjectNetDataset(VisionDataset):
    """
    ObjectNet dataset. Implementation from https://github.com/fairinternal/ssl-3d/blob/f838ce35aeace6b2d7fe337b5ff09cafad581cb5/moco_dist/eval_objectnet.py.
    Args:
        path (string): path directory where images are downloaded to. The images can be grouped in folders. (the folder structure will be ignored)
        augmentations (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, 'augmentations.ToTensor'
    """

    def __init__(self, path, augmentations=None):
        """Init ObjectNet pytorch dataloader."""
        super().__init__(path, augmentations)
        self.augmentations = augmentations

        self.data_path = os.path.join(path, "images")

        # from https://github.com/lucaslie/torchprune/blob/14b392cb3e5523c8677bcb78c92d5e155366d85a/src/torchprune/torchprune/util/datasets/objectnet.py
        o_dataset = ImageFolder(self.data_path)

        # get mappings folder
        mappings_folder = os.path.join(path, "mappings")

        # get ObjectNet label to ImageNet label mapping
        with open(os.path.join(mappings_folder, "objectnet_to_imagenet_1k.json")) as f:
            o_label_to_all_i_labels = json.load(f)

        # now remove double i labels to avoid confusion
        o_label_to_i_labels = {
            o_label: all_i_label.split("; ")
            for o_label, all_i_label in o_label_to_all_i_labels.items()
        }

        # some in-between mappings ...
        o_folder_to_o_idx = o_dataset.class_to_idx
        with open(os.path.join(mappings_folder, "folder_to_objectnet_label.json")) as f:
            o_folder_o_label = json.load(f)

        # now get mapping from o_label to o_idx
        o_label_to_o_idx = {
            o_label: o_folder_to_o_idx[o_folder]
            for o_folder, o_label in o_folder_o_label.items()
        }

        # some in-between mappings ...
        with open(
            os.path.join(mappings_folder, "pytorch_to_imagenet_2012_id.json")
        ) as f:
            i_idx_to_i_line = json.load(f)
        with open(os.path.join(mappings_folder, "imagenet_to_label_2012_v2")) as f:
            i_line_to_i_label = f.readlines()

        i_line_to_i_label = {
            i_line: i_label[:-1] for i_line, i_label in enumerate(i_line_to_i_label)
        }

        # now get mapping from i_label to i_idx
        i_label_to_i_idx = {
            i_line_to_i_label[i_line]: int(i_idx)
            for i_idx, i_line in i_idx_to_i_line.items()
        }

        # now get the final mapping of interest!!!
        o_idx_to_i_idxs = {
            o_label_to_o_idx[o_label]: [
                i_label_to_i_idx[i_label] for i_label in i_labels
            ]
            for o_label, i_labels in o_label_to_i_labels.items()
        }

        # now get a list of files of interest
        overlapping_samples = []
        for filepath, o_idx in o_dataset.samples:
            if o_idx not in o_idx_to_i_idxs:
                continue
            rel_file = os.path.relpath(filepath, self.data_path)
            overlapping_samples.append((rel_file, o_idx_to_i_idxs[o_idx][0]))

        self.samples = overlapping_samples  # list of tuples (rel_img_path, label idx))

    def __getitem__(self, index):
        """
        Get an image and its label.
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the image file name
        """
        img, target = self.get_image(index)
        if self.augmentations is not None:
            img = self.augmentations(img)
        return img, target

    def get_image(self, index):
        """
        Load the image and its label.
        Args:
            index (int): Index
        Return:
            tuple: Tuple (image, target). target is the image file name
        """
        img_path, label = self.samples[index]
        img = self.pil_loader(os.path.join(self.data_path, img_path))

        # crop out red border
        width, height = img.size
        cropArea = (2, 2, width - 2, height - 2)
        img = img.crop(cropArea)
        return (img, label)

    def __len__(self):
        """Get the number of ObjectNet images to load."""
        return len(self.samples)

    def pil_loader(self, path):
        """Pil image loader."""
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, "rb") as f:
            img = Image.open(f)
            return img.convert("RGB")


class ObjectNetDataModule(ImageDataModule):
    def __init__(
        self,
        data_dir: str = "data/objectnet",
        batch_size: int = 32,
        num_workers=8,
        image_size=224,
    ):
        """Pytorch lightning based datamodule for ObjectNet dataset, found here: https://objectnet.dev/

        Args:
            data_dir (str, optional): Path to imagenet dataset directory. Defaults to "/checkpoint/meganrichards/datasets/objectnet/".
            batch_size (int, optional): Batch size to use in datamodule. Defaults to 32.
            num_workers (int, optional): Number of workers to use in the dataloaders. Defaults to 8.
            image_size (int, optional): Side length for image crop. Defaults to 224.
        """

        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            image_size=image_size,
            num_workers=num_workers,
        )

    def _get_dataset(self, path, stage, augmentations):
        return ObjectNetDataset(path, augmentations)
