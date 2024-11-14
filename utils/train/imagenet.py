# import os

from ..dataset import ImageNet, load_dataloader
from ..typing import DataLoader, LightningModule, Module, Optional, Path, Trainer
from .train import Wrapper, load_trainer

__all__ = [
    "ImageNetTrainer",
]


class ImageNetTrainer:
    def __init__(
        self,
        dataset_dir: Path,
        save_dir: Path,
        name: str,
        version: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.trainer: Trainer = load_trainer(save_dir, name, version, **kwargs)
        self.log_dir = self.trainer._log_dir

        self.train_dataloader: DataLoader = load_dataloader(
            dataset=ImageNet(
                root=dataset_dir,
                split="train",
                transform="auto",
                **kwargs,
            ),
            train=True,
            **kwargs,
        )
        self.val_dataloader: DataLoader = load_dataloader(
            dataset=ImageNet(
                root=dataset_dir,
                split="val",
                transform="auto",
                **kwargs,
            ),
            train=False,
            **kwargs,
        )
        self.model: LightningModule

    def train(
        self,
        model: Module,
        resume: bool = False,
        **kwargs,
    ) -> None:
        model = Wrapper(model, **kwargs)
        self.trainer.fit(
            model,
            self.train_dataloader,
            self.val_dataloader,
            # ckpt_path=(
            #     os.path.join(self.log_dir, "checkpoints", "last.ckpt")
            #     if resume
            #     else None
            # ),
            ckpt_path="last" if resume else None,
        )
