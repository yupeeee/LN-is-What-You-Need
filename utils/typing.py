import os
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from lightning import LightningModule, Trainer
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "Any",
    "Dataset",
    "DataLoader",
    "Devices",
    "Dict",
    "LightningModule",
    "List",
    "Literal",
    "Module",
    "Optimizer",
    "Optional",
    "Path",
    "Sequence",
    "Tensor",
    "Trainer",
    "Tuple",
    "LRScheduler",
]

Devices = Union[int, str, List[Union[int, str]]]
Path = Union[str, bytes, os.PathLike]
