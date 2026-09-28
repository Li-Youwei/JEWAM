# Training and ablations

[Documentation index](README.md) · [Project overview](../README.md)

Run all commands from the repository root in an activated environment.

The full model uses batch 128, BF16, AdamW (`lr=5e-5`, `weight_decay=0.05`),
100,000 steps, 2,000 warmup steps followed by cosine decay, gradient clipping
1.0, and label smoothing 0.1. Keep one GPU for BatchNorm/SIGReg runs.

The runner defaults to the full method with a 9:1 split. It saves checkpoints
every 4,000 training batches (also steps with the default accumulation of 1),
ranks them by task-balanced validation CE, and evaluates the best three, each
with 50 rollouts per task. It does not automatically select by rollout success.

```bash
CUDA_VISIBLE_DEVICES=0 ARM=full SEED=3072 RUN_NAME=full_split \
  bash scripts/train_eval.sh
```

After fixing the training budget using validation, retrain on **all**
demonstrations for the main-result protocol. This saves and evaluates the final
100,000-step object checkpoint without validation-based selection:

```bash
CUDA_VISIBLE_DEVICES=0 ARM=full SEED=3072 TRAIN_SPLIT=1 RUN_NAME=full_all_data \
  bash scripts/train_eval.sh
```

Use a new `RUN_NAME` for each experiment. See `bash scripts/train_eval.sh --help`
for path, retention, and evaluation overrides. `CKPT_SAVE_TOP_K=-1` retains all
checkpoints; they can occupy substantial disk space. The runner defaults to
offline Hugging Face access, so cache assets first.

To train without launching evaluation:

```bash
python -m jewam.training.train --config-name=jewam
# Inspect the resolved configuration without loading data or model weights:
python -m jewam.training.train --config-name=jewam --cfg job --resolve
# Full-data training only:
python -m jewam.training.train --config-name=jewam train_split=1 \
  subdir=full_all_data +trainer.limit_val_batches=0
```

`config/train/base.yaml` supplies shared defaults and remains the default
for bare `python -m jewam.training.train` calls. Select `--config-name=jewam`
for the full model as shown above. `config/train/overfit.yaml` is a small-data
pipeline diagnostic.

Checkpoints use the `jewam` filename prefix, and TensorBoard events are written
under `tb_logs/jewam/` within each run directory. Use `JEWAM_HOME` to override
the checkpoint root for direct training.

## Paper ablations

All ablations use `TRAIN_SPLIT=0.9`, with the same seed and data ordering.
Set the following environment variables before `bash scripts/train_eval.sh`;
use a distinct `RUN_NAME` each time.

| Configuration | Runner settings |
|---|---|
| Full model | `ARM=full` |
| w/o SIGReg | `ARM=full SIGREG_WEIGHT=0` |
| w/o MoT | `ARM=all4_sp_sigreg_dinov2_frozen_visual17_patch_sp` |
| w/o SP + SIGReg | `ARM=all4_dinov2_frozen_visual17_mot` |
| w/o JEPA | `ARM=all4_sp_dinov2_frozen_visual17_patch_sp_mot_dino768_identity` |
| w/o FAST | `ARM=full ACTION_CODEC=worldvla_bins` |

The no-JEPA variant uses 768D backbone features and disables SIGReg. The no-FAST
variant uses 256 scalar bins and 140 action tokens. Prepare its data separately:

```bash
ACTION_CODEC=worldvla_bins OUT_ROOT=data/libero_processed_bins \
  bash scripts/preprocess_all4.sh
python -m scripts.data.make_flat_dataset --processed-root data/libero_processed_bins
ARM=full ACTION_CODEC=worldvla_bins \
  FLAT_DIR=data/libero_processed_bins/all4_flat \
  PROCESSED_ROOT=data/libero_processed_bins RUN_NAME=without_fast \
  bash scripts/train_eval.sh
```

Other legacy ARM names and optional model switches remain for checkpoint/config
compatibility; they are not recommended as the paper's final model.
