# Data and pretrained assets

[Documentation index](README.md) · [Project overview](../README.md)

Run all commands from the repository root in an activated environment.

Use this local layout, or override paths with environment variables:

```text
data/
  libero_raw/{libero_spatial,libero_object,libero_goal,libero_10}/*.hdf5
  fast_tokenizer/                 Fitted tokenizer, config, and processor code
  libero_processed/
    {libero_spatial,libero_object,libero_goal,libero_10}/*.h5
    all4_flat/*.h5                Links to the 40 processed tasks
pretrained/dinov2-base/           Hugging Face DINOv2-base snapshot
checkpoints/<run-name>/           Object checkpoints, config.yaml, and tb_logs/
outputs/eval_videos/              Optional rollout videos
```

All these artifact directories are ignored by Git. Obtain the original LIBERO
HDF5 demonstrations from the [official dataset instructions](https://github.com/Lifelong-Robot-Learning/LIBERO#datasets).
Keep raw files read-only. LIBERO-10 is the ten-task subset of LIBERO-100; arrange
its task files under `libero_10/`.

Before offline runs, download/cache the frozen encoders and the
[FAST processor](https://huggingface.co/physical-intelligence/fast):

```bash
python - <<'PY'
from huggingface_hub import snapshot_download
from transformers import AutoProcessor, T5EncoderModel, T5Tokenizer

snapshot_download('facebook/dinov2-base', local_dir='pretrained/dinov2-base')
T5Tokenizer.from_pretrained('t5-small')
T5EncoderModel.from_pretrained('t5-small')
AutoProcessor.from_pretrained('physical-intelligence/fast', trust_remote_code=True)
PY
```

Fit the shared tokenizer, preprocess each task, and build the flat directory:

```bash
python -m scripts.data.fit_tokenizer_all4 \
  --raw-root data/libero_raw \
  --tokenizer-out data/fast_tokenizer \
  --audit-out data/libero_processed/audit_token_length.json
bash scripts/preprocess_all4.sh
python -m scripts.data.make_flat_dataset
python -m scripts.diagnostics.check_fast_roundtrip \
  --processed-dir data/libero_processed/libero_spatial \
  --tokenizer data/fast_tokenizer
python -m scripts.diagnostics.verify_sampler_balance --hdf5-dir data/libero_processed/all4_flat
```

Preprocessing uses stride 1, horizon 20, observation at `t`, future state at
`t+20`, and pose targets at `t+k+1`. Task-specific 1st/99th percentiles normalize
actions. The tokenizer is fitted jointly across all four suites; use the same
fitted tokenizer and saved HDF5 normalization bounds for training and evaluation.

**Data protocol:** preprocessing computes normalization and fits
FAST on the supplied task demonstrations *before* the training/validation split.
It does not fit preprocessing statistics on a train-only subset. Demonstration
splitting then occurs in
[`jewam/training/train.py`](../jewam/training/train.py), with no chunks shared
across the two sets.
The split is global over sorted `(task file, demo id)` pairs, not separately
stratified within each task. Preserve the original flat filenames/order, data,
and seed when reproducing an existing split. The linking helper preserves task
basenames and rejects collisions; it cannot recover historical filenames.
