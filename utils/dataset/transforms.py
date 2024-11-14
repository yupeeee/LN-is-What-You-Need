# https://github.com/pytorch/vision/blob/main/references/classification/presets.py
# https://github.com/pytorch/vision/blob/main/references/classification/transforms.py

import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.functional import InterpolationMode

__all__ = [
    "get_mixup_cutmix",
    "ClassificationPresetTrain",
    "ClassificationPresetEval",
]


def get_mixup_cutmix(
    *,
    mixup_alpha,
    cutmix_alpha,
    num_classes,
):
    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(T.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        mixup_cutmix.append(T.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    if not mixup_cutmix:
        return None

    return T.RandomChoice(mixup_cutmix)


class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(
        self,
        *,
        crop_size,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        hflip_prob=0.5,
        auto_augment_policy=None,
        ra_magnitude=9,
        augmix_severity=3,
        random_erase_prob=0.0,
        backend="pil",
    ):
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(
            T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True)
        )
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(
                    T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude)
                )
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(
                    T.AugMix(interpolation=interpolation, severity=augmix_severity)
                )
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(
                    T.AutoAugment(policy=aa_policy, interpolation=interpolation)
                )

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms.extend(
            [
                T.ToDtype(torch.float, scale=True),
                T.Normalize(mean=mean, std=std),
            ]
        )
        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))

        transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(
        self,
        *,
        crop_size,
        resize_size=256,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation=InterpolationMode.BILINEAR,
        backend="pil",
    ):
        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms += [
            T.ToDtype(torch.float, scale=True),
            T.Normalize(mean=mean, std=std),
        ]

        transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)
