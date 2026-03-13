"""
transforms.py
-------------
Augmentation pipelines for training, validation, and test splits.

Uses `albumentations` for image-level augmentations and `torchvision`
for the final tensor conversion.  Two factory functions are exposed:

* ``get_train_transforms`` — heavy augmentation for the training set.
* ``get_val_test_transforms`` — deterministic resizing + normalisation.

The normalisation statistics follow ImageNet conventions because all
backbone models (EfficientNet-B3, etc.) are pre-trained on ImageNet.
"""

from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2


# ImageNet normalisation constants
_IMAGENET_MEAN: Tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: Tuple[float, float, float] = (0.229, 0.224, 0.225)


def get_train_transforms(image_size: int = 300) -> A.Compose:
    """Return a heavy augmentation pipeline suitable for training.

    Includes spatial and colour augmentations that are commonly used for
    dermoscopic image analysis:
    - Random horizontal / vertical flips
    - Random rotation (±45°)
    - Random brightness / contrast / saturation / hue shifts
    - Gaussian noise injection (std drawn uniformly from [0.01, 0.05])
    - Random crop / resize combination
    - Optional coarse dropout (simulates occlusion)
    - ImageNet normalisation + conversion to ``torch.Tensor``

    Parameters
    ----------
    image_size : int
        Target spatial resolution (height == width).  Default is 300
        to match EfficientNet-B3's native input size.

    Returns
    -------
    albumentations.Compose
        An albumentations pipeline that accepts a NumPy HWC image and
        returns a CHW float32 PyTorch tensor.
    """
    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.7, 1.0),
                ratio=(0.75, 1.333),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=45, p=0.5),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5,
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.3),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=5),
                    A.MedianBlur(blur_limit=5),
                    A.GaussianBlur(blur_limit=5),
                ],
                p=0.2,
            ),
            A.CoarseDropout(
                num_holes_range=(1, 4),
                hole_height_range=(16, 48),
                hole_width_range=(16, 48),
                fill=0,
                p=0.2,
            ),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def get_val_test_transforms(image_size: int = 300) -> A.Compose:
    """Return a deterministic pipeline for validation and test splits.

    Only performs resize and normalisation — no stochastic augmentations.

    Parameters
    ----------
    image_size : int
        Target spatial resolution (height == width).  Default is 300.

    Returns
    -------
    albumentations.Compose
        An albumentations pipeline that accepts a NumPy HWC image and
        returns a CHW float32 PyTorch tensor.
    """
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
            ToTensorV2(),
        ]
    )
