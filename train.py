import argparse
import os

from paths import IMAGENET_DIR, LOGS_DIR
from utils import (
    ImageNetTrainer,
    load_config,
    load_model,
    replace_bn_with_ln,
    replace_ln_with_bn,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--bn2ln", action="store_true", default=False)
    parser.add_argument("--ln2bn", action="store_true", default=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", action="store_true", default=False)
    args = parser.parse_args()

    config = load_config(os.path.join("cfgs", f"{args.config}.yaml"))

    model = load_model(args.model)
    if args.bn2ln:
        model = replace_bn_with_ln(model)
    elif args.ln2bn:
        model = replace_ln_with_bn(model)

    trainer = ImageNetTrainer(
        dataset_dir=IMAGENET_DIR,
        save_dir=LOGS_DIR,
        name=args.model + ("_ln" if args.bn2ln else "_bn" if args.ln2bn else ""),
        version=args.config,
        **config,
    )
    trainer.train(
        model=model,
        resume=args.resume,
        **config,
    )
