import os

import torchvision

from ..typing import Dataset, Literal, Path
from .transforms import ClassificationPresetEval, ClassificationPresetTrain

__all__ = [
    "ImageNet",
]


def ImageNet(
    root: Path,
    split: Literal["train", "val"],
    transform: Literal["auto", "train", "eval"] = "auto",
    **kwargs,
) -> Dataset:
    crop_size = kwargs.get("crop_size", 224 if split == "train" else 256)
    resize_size = kwargs.get("resize_size", 224)
    interpolation = kwargs.get("interpolation", "bilinear")
    auto_augment_policy = kwargs.get("auto_augment", None)
    random_erase_prob = kwargs.get("random_erase", 0.0)
    ra_magnitude = kwargs.get("ra_magnitude", None)
    augmix_severity = kwargs.get("augmix_severity", None)
    backend = kwargs.get("backend", "pil")

    root = os.path.join(root, split)
    if transform == "auto":
        transform = "train" if split == "train" else "eval"

    dataset = torchvision.datasets.ImageFolder(
        root=root,
        transform=(
            ClassificationPresetTrain(
                crop_size=crop_size,
                interpolation=interpolation,
                auto_augment_policy=auto_augment_policy,
                random_erase_prob=random_erase_prob,
                ra_magnitude=ra_magnitude,
                augmix_severity=augmix_severity,
                backend=backend,
            )
            if transform == "train"
            else ClassificationPresetEval(
                crop_size=crop_size,
                resize_size=resize_size,
                interpolation=interpolation,
                backend=backend,
            )
        ),
    )

    return dataset
