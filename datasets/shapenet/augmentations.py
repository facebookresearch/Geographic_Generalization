"""
Defines frame augmentations used during training
"""
import numpy as np
from typing import List, Callable

from pl_bolts.utils import _OPENCV_AVAILABLE, _TORCHVISION_AVAILABLE
from pl_bolts.utils.warnings import warn_missing_pkg
from pl_bolts.transforms.dataset_normalizations import imagenet_normalization

if _TORCHVISION_AVAILABLE:
    from torchvision import transforms
else:  # pragma: no cover
    warn_missing_pkg("torchvision")

if _OPENCV_AVAILABLE:
    import cv2
else:  # pragma: no cover
    warn_missing_pkg("cv2", pypi_name="opencv-python")

import torchvision.transforms as torch_transforms
import torchvision.transforms.functional as F
import torch


class DetRandomResizedCrop(torch_transforms.RandomResizedCrop):
    def forward(self, img, params=None):
        """
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image + PARAMS
        """
        if params is None:
            i, j, h, w = self.get_params(img, self.scale, self.ratio)
        else:
            i, j, h, w = params
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), (
            i,
            j,
            h,
            w,
        )


class DetRandomHorizontalFlip(torch_transforms.RandomHorizontalFlip):
    def forward(self, img, params=None):
        """
        Args:
            img (PIL Image or Tensor): Image to be flipped.

        Returns:
            PIL Image or Tensor: Randomly flipped image + PARAMS
        """
        flipped = None
        if params is None:
            if torch.rand(1) < self.p:
                flipped = True
                img_ = F.hflip(img)
            else:
                flipped = False
                img_ = img
        else:
            flipped = params
            if flipped:
                img_ = F.hflip(img)
            else:
                img_ = img

        return img_, flipped


class DetColorJitter(torch_transforms.ColorJitter):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def forward(self, img, params=None):

        if params is None:
            apply = False
            if self.p < torch.rand(1):
                apply = False
            else:
                apply = True
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = self.get_params(
                self.brightness, self.contrast, self.saturation, self.hue
            )
        else:
            apply = params[0]
            (
                fn_idx,
                brightness_factor,
                contrast_factor,
                saturation_factor,
                hue_factor,
            ) = params[1:]

        if apply:
            for fn_id in fn_idx:
                if fn_id == 0 and brightness_factor is not None:
                    img = F.adjust_brightness(img, brightness_factor)
                elif fn_id == 1 and contrast_factor is not None:
                    img = F.adjust_contrast(img, contrast_factor)
                elif fn_id == 2 and saturation_factor is not None:
                    img = F.adjust_saturation(img, saturation_factor)
                elif fn_id == 3 and hue_factor is not None:
                    img = F.adjust_hue(img, hue_factor)

        return img, (
            apply,
            fn_idx,
            brightness_factor,
            contrast_factor,
            saturation_factor,
            hue_factor,
        )


class DetGaussianBlur(torch_transforms.GaussianBlur):
    def __init__(self, p=0.5, **kwargs):
        super().__init__(**kwargs)
        self.p = p

    def forward(self, img, params=None):

        if params is None:
            apply = False
            if self.p < torch.rand(1):
                apply = False
            else:
                apply = True
            sigma = self.get_params(self.sigma[0], self.sigma[1])
        else:
            apply = params[0]
            sigma = params[1]
        if apply:
            return F.gaussian_blur(img, self.kernel_size, [sigma, sigma]), (
                apply,
                sigma,
            )
        else:
            return img, (apply, sigma)


class DetRandomApply(torch_transforms.RandomApply):
    """
    Apply randomly or deterministically (depending on params) a list of transformations with a given probability.
    """

    def forward(self, img, params=None):

        apply = False
        # Get if apply or not
        if params is None:
            if self.p < torch.rand(1):
                apply = False
            else:
                apply = True
        else:
            apply = params

        if apply:
            img_ = img
            for t in self.transforms:
                img_ = t(img_)
        else:
            img_ = img
        return img_, apply


class CVLRTrainDataTransform:
    """Transforms for CVLR
    Transform::
        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()
    """

    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        kernel_size = int(0.1 * self.input_height)
        if kernel_size % 2 == 0:
            kernel_size += 1

        data_transforms = [
            transforms.RandomResizedCrop(
                size=self.input_height, scale=(0.3, 1), ratio=(0.5, 2)
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),
            # following transforms require size [...,C,H,W]
            transforms.RandomApply(
                [self.color_jitter], p=0.8
            ),  # torchvision uses random order, values are from tf code
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=(0.1, 2)),
            transforms.Lambda(lambda x: x.permute(1, 0, 2, 3)),
        ]

        self.train_transform = transforms.Compose(
            data_transforms,
        )

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)

        return xi


class SimCLRTrainDataTransform:
    """Transforms for SimCLR using a modifications of Lightning Bolts
    Transform::
        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform
        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        add_to_tensor: bool = False,
        add_imagenet_normalization: bool = False,
    ) -> None:

        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `transforms` from `torchvision` which is not installed yet."
            )

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.add_to_tensor = add_to_tensor
        self.add_imagenet_normalization = add_imagenet_normalization

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        data_transforms = self.append_transforms(
            data_transforms, gaussian_blur=self.gaussian_blur
        )

        self.train_transform = transforms.Compose(
            data_transforms,
        )

        # add online train transform of the size of global view
        online_data_transforms = [
            transforms.RandomResizedCrop(self.input_height),
            transforms.RandomHorizontalFlip(),
        ]
        online_data_transforms = self.append_transforms(
            online_data_transforms, gaussian_blur=False
        )
        self.online_transform = transforms.Compose(online_data_transforms)

    def append_transforms(
        self, data_transforms: List[Callable], gaussian_blur: bool = False
    ) -> List[Callable]:
        if gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1
            data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))

        if self.add_to_tensor:
            data_transforms.append(transforms.ToTensor())

        if self.add_imagenet_normalization:
            data_transforms.append(imagenet_normalization())
        return data_transforms

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj, self.online_transform(sample)


class SimCLREvalDataTransform(SimCLRTrainDataTransform):
    """Transforms for SimCLR.
    Transform::
        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform
        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self,
        input_height: int = 224,
        gaussian_blur: bool = True,
        jitter_strength: float = 1.0,
        add_to_tensor: bool = False,
        add_imagenet_normalization: bool = False,
    ):
        super().__init__(
            input_height=input_height,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength,
            add_to_tensor=add_to_tensor,
            add_imagenet_normalization=add_imagenet_normalization,
        )

        # replace online transform with eval time transform
        eval_transforms = [
            transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
            transforms.CenterCrop(self.input_height),
        ]
        eval_transforms = self.append_transforms(eval_transforms, gaussian_blur=False)

        self.online_transform = transforms.Compose(eval_transforms)


class GaussianBlur:
    """From https://github.com/PyTorchLightning/lightning-bolts/blob/master/pl_bolts/models/self_supervised/simclr/transforms.py"""

    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):
        if not _TORCHVISION_AVAILABLE:  # pragma: no cover
            raise ModuleNotFoundError(
                "You want to use `GaussianBlur` from `cv2` which is not installed yet."
            )

        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(
                sample, (self.kernel_size, self.kernel_size), sigma
            )

        return sample


class SimCLRFinetuneTransform:
    def __init__(
        self,
        input_height: int = 224,
        jitter_strength: float = 1.0,
        normalize=None,
        eval_transform: bool = False,
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        if not eval_transform:
            data_transforms = [
                transforms.RandomResizedCrop(size=self.input_height),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([self.color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
            ]
        else:
            data_transforms = [
                transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
                transforms.CenterCrop(self.input_height),
            ]

        if normalize is None:
            final_transform = transforms.ToTensor()
        else:
            final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        data_transforms.append(final_transform)
        self.transform = transforms.Compose(data_transforms)

    def __call__(self, sample):
        return self.transform(sample)
