import timm

from ..typing import Module

__all__ = [
    "load_model",
]


def load_model(
    model_name: str,
    num_classes: int = 1000,
    image_size: int = 224,
    num_channels: int = 3,
) -> Module:
    kwargs = {
        "model_name": model_name,
        "pretrained": False,
        "num_classes": num_classes,
        "img_size": image_size,
        "in_chans": num_channels,
    }

    # Some models require `img_size`...
    try:
        model = timm.create_model(**kwargs)
    # ...and some do not
    except TypeError as e:
        _ = kwargs.pop("img_size")
        model = timm.create_model(**kwargs)

    return model
