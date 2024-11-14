import argparse
import os

from paths import IMAGENET_DIR, LOGS_DIR
from utils import ImageNetTrainer, load_config, load_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config = load_config(os.path.join("cfgs", args.config))
    model = load_model(args.model)
    trainer = ImageNetTrainer(
        dataset_dir=IMAGENET_DIR,
        save_dir=LOGS_DIR,
        name=args.model,
        version=args.config,
        **config,
    )
    trainer.train(
        model=model,
        resume=args.resume,
        **config,
    )
