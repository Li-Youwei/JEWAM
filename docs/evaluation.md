# Evaluation

[Documentation index](README.md) · [Project overview](../README.md)

Run all commands from the repository root in an activated environment.

Use trusted **object checkpoints** for the full method and ablations with
MoT, state prediction, or visual patch tokens. The weights-only reconstruction
path supports the legacy CLS-only baseline, not the full architecture.

```bash
python -m jewam.evaluation.libero \
  --checkpoint checkpoints/full_all_data/jewam_step_100000_object.ckpt \
  --tokenizer data/fast_tokenizer \
  --processed-dir data/libero_processed/libero_spatial \
  --suite libero_spatial --num-episodes 50 --seed 3072 --device cuda
```

Repeat for `libero_object`, `libero_goal`, and `libero_10`, using the matching
processed directory. Defaults use 10 settling steps and rollout horizons
220 / 280 / 300 / 520 respectively. Use `--task-id 0` for one task and
`--save-videos` for recordings. `--help` works without importing the simulator.

To evaluate an existing runner directory across all four suites:

```bash
EVAL_ONLY=true ARM=full TRAIN_SPLIT=1 RUN_NAME=full_all_data \
  bash scripts/train_eval.sh
python -m scripts.eval.aggregate_all4_results \
  --ckpt-dir checkpoints/full_all_data \
  --checkpoint-stem jewam_step_100000_object
```

For a split run, omit `TRAIN_SPLIT=1`; checkpoint selection requires its original
TensorBoard validation logs. To evaluate a particular checkpoint without those
logs, use `python -m jewam.evaluation.libero` directly. The aggregation command
names the exact checkpoint so results from different candidates are not
combined. With 50 rollouts for every task, the pooled rate equals the mean of
the four suite rates.

## Checkpoint compatibility

Object checkpoints store Python class paths. The root modules `jepa.py`,
`module.py`, and `vision_backbone.py` support checkpoints that reference those
module names; keep them available when loading such files. Checkpoints with
`jewam.*` class paths require a repository version that provides those modules.

Checkpoint selection accepts different filename prefixes and TensorBoard log
directory names. Keep the original validation logs with each split-run
checkpoint directory.
