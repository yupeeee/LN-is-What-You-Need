# LN-is-What-You-Need

## Model Training

To train a model (e.g., `swin_tiny_patch4_window7_224`) on ImageNet-1K, run the following command:
```bash
python train.py --model TIMM_MODEL_NAME --config PATH_TO_YAML_CONFIG
```

To train a model with BN replaced by LN (or vice versa), run the following command:
```bash
python train.py --model TIMM_MODEL_NAME --config PATH_TO_YAML_CONFIG --bn2ln
python train.py --model TIMM_MODEL_NAME --config PATH_TO_YAML_CONFIG --ln2bn
```

To resume training, add the `--resume` flag:
```bash
python train.py --model TIMM_MODEL_NAME --config PATH_TO_YAML_CONFIG --resume
```

### YAML Configuration

The YAML configuration file is used to specify the hyperparameters for training.
You can find some example configurations in the `cfgs` directory.

```YAML
--- # cfgs/swin_sbatch.yaml

devices: gpu:0,1,2,3
seed: 0
amp: true
save_per_epoch: 1

#
# Dataset
#
num_classes: 1000
batch_size: 64
num_workers: 8
pin_memory: true

#
# Regularization
#
label_smoothing: 0.1
mixup_alpha: 0.8
cutmix_alpha: 1.0
gradient_clip_val: 5.0

#
# Training hyperparameters
#
epochs: 300
init_lr: 1.e-3

optimizer: AdamW
optimizer_cfg:
  weight_decay: 5.e-2

lr_scheduler: CosineAnnealingLR
lr_scheduler_cfg:
  eta_min: 1.e-5

warmup_lr_scheduler: LinearLR
warmup_lr_scheduler_cfg:
  start_factor: 1.e-2
  total_iters: 20

criterion: CrossEntropyLoss

... # end
```