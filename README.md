# JEWAM

**Joint-Embedding World Action Model** is the repository name for the method in
*Autoregressive World Action Model with a Joint-Embedding Predictive Architecture*.
The policy learns robot manipulation from the 40 tasks in four LIBERO suites,
without robot-data pretraining. It predicts FAST action tokens from two RGB
views, language, and proprioception. Future-state prediction supervises training;
rollouts use the action branch only.

This release preserves the final `codex/bestmodel` implementation. It includes
the full method and the components needed for its ablations. Pretrained policy
checkpoints, demonstrations, and manuscript files are not included.

## Method

- **Inputs:** agent and wrist RGB images resized to 224 × 224, frozen T5-small
  language features (at most 25 tokens), and 9D proprioception:
  position (3), xyzw quaternion (4), and both gripper finger positions (2).
- **Visual representation:** frozen DINOv2-base; CLS plus 16 pooled patches per
  view. Separate 768 → 2048 → 192 MLPs project CLS and patch features, using
  BatchNorm and LayerNorm respectively.
- **Predictor:** six MoT layers with global attention and modality-specific
  parameters; 192D tokens, 16 heads, 64D per head, 2048D FFNs, dropout 0.2.
- **Actions:** a shared 1,024-entry FAST vocabulary encodes 20-step, 7D chunks
  into at most 80 tokens. Translation and rotation targets are relative to the
  pose at the start of the chunk; the seventh dimension is the gripper command.
- **Training:** action CE + 1.0 × future-state loss + 0.1 × SIGReg. Each visual
  loss averages CLS MSE and patch MSE with equal weight. Future targets retain
  projector gradients; SIGReg acts on current/future CLS features only.
- **Execution:** greedily decode an action chunk, then execute all 20 targets.
  Each low-level command uses the measured current pose to correct the residual
  to the anchor-relative target. Replan after the chunk or episode termination.

The paper reports 85.30% mean success for the all-demonstration main result
(Spatial 93.2, Object 97.8, Goal 66.8, LIBERO-10 83.4). The 84.25% full-model
ablation reference uses a separate 9:1 demonstration split; these are different
runs. These reported results have not been rerun as part of repository cleanup.

## Layout

```text
config/train/             Hydra base, full JEWAM, overfit, and dataset configs
scripts/
  preprocess_all4.sh      Preprocess all four suites, preserving raw data
  make_flat_dataset.py    Build the flat training directory with symlinks
  train_eval.sh           Train, select checkpoints, and evaluate four suites
tests/                    CPU regression and synthetic pipeline smoke tests
jepa.py                   Policy container and visual/language encoding
module.py                 MoT/shared transformer, projectors, and SIGReg
vision_backbone.py        Frozen Hugging Face vision backbone adapter
train.py                  Lightning training and joint losses
eval_libero.py            Closed-loop LIBERO evaluation
libero_dataset.py         HDF5 dataset and balanced sampler weights
preprocess_libero.py       Single-task preprocessing
fit_tokenizer_all4.py      Shared FAST tokenizer fitting
fast_utils.py             FAST decoding and action denormalization
action_codec.py           FAST and uniform-bin ablation interfaces
utils.py                  Object checkpoints and validation metrics
pick_best_ckpt.py          Task-balanced validation CE checkpoint ranking
aggregate_all4_results.py  Per-checkpoint rollout summaries
check_fast_roundtrip.py    Action codec/data consistency check
verify_sampler_balance.py Task/demo/chunk sampling check
requirements*.txt          Core, training, evaluation, and test dependencies
environment.yml           Python 3.10 training environment
```

Core Python modules intentionally remain at the repository root: existing
`_object.ckpt` files record paths such as `jepa.JEPA`, `module.ARPredictor`, and
`vision_backbone.HFVisionBackbone`. The `lewm` checkpoint filename prefix and
base config name remain compatible with saved experiments.

## Installation

Use **Python 3.10**. Full training and rollout evaluation target Linux with an
NVIDIA GPU; CPU tests and CLI checks can run on macOS. Activate your environment
before running scripts; they do not assume a particular conda installation.

```bash
git clone https://github.com/Li-Youwei/JEWAM.git
cd JEWAM
conda env create -f environment.yml
conda activate jewam
```

Alternatively, in an existing Python 3.10 environment:

```bash
python -m pip install -r requirements.txt
```

Install a matching PyTorch/torchvision CUDA build for your machine. For core
CPU checks only, use `python -m pip install -r requirements-dev.txt`.
`stable-pretraining==0.1.6` is an API compatibility pin: later releases change
`Manager.ckpt_path` semantics. The dependency files are release compatibility
specifications, **not an archived lockfile from the original training machine**.

For rollout evaluation, install the simulation dependencies and
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO):

```bash
python -m pip install -r requirements.txt -r requirements-eval.txt
mkdir -p third_party
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
python -m pip install -e third_party/LIBERO --no-deps
```

The evaluation requirements constrain the legacy simulator's NumPy/PyTorch
compatibility. Do not install LIBERO's old full requirements over this
project's Transformers dependency. Configure LIBERO's asset, BDDL, and initial
state paths as described upstream. For headless Linux rendering, the evaluator
defaults to `MUJOCO_GL=egl`; it respects an existing value.

## Data and pretrained assets

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
python fit_tokenizer_all4.py \
  --raw-root data/libero_raw \
  --tokenizer-out data/fast_tokenizer \
  --audit-out data/libero_processed/audit_token_length.json
bash scripts/preprocess_all4.sh
python scripts/make_flat_dataset.py
python check_fast_roundtrip.py \
  --processed-dir data/libero_processed/libero_spatial \
  --tokenizer data/fast_tokenizer
python verify_sampler_balance.py --hdf5-dir data/libero_processed/all4_flat
```

Preprocessing uses stride 1, horizon 20, observation at `t`, future state at
`t+20`, and pose targets at `t+k+1`. Task-specific 1st/99th percentiles normalize
actions. The tokenizer is fitted jointly across all four suites; use the same
fitted tokenizer and saved HDF5 normalization bounds for training and evaluation.

**Data protocol:** the preserved implementation computes normalization and fits
FAST on the supplied task demonstrations *before* the training/validation split.
It does not fit preprocessing statistics on a train-only subset. Demonstration
splitting then occurs in `train.py`, with no chunks shared across the two sets.
The split is global over sorted `(task file, demo id)` pairs, not separately
stratified within each task. Preserve the original flat filenames/order, data,
and seed when reproducing an existing split. The linking helper preserves task
basenames and rejects collisions; it cannot recover historical filenames.

## Training

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
python train.py --config-name=jewam
# Inspect the resolved configuration without loading data or model weights:
python train.py --config-name=jewam --cfg job --resolve
# Full-data training only:
python train.py --config-name=jewam train_split=1 \
  subdir=full_all_data +trainer.limit_val_batches=0
```

`config/train/lewm.yaml` remains a compatible base, not the paper full-model
preset. `config/train/overfit.yaml` is a small-data pipeline diagnostic.

### Paper ablations

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
python scripts/make_flat_dataset.py --processed-root data/libero_processed_bins
ARM=full ACTION_CODEC=worldvla_bins \
  FLAT_DIR=data/libero_processed_bins/all4_flat \
  PROCESSED_ROOT=data/libero_processed_bins RUN_NAME=without_fast \
  bash scripts/train_eval.sh
```

Other legacy ARM names and optional model switches remain for checkpoint/config
compatibility; they are not recommended as the paper's final model.

## Evaluation

Use trusted **object checkpoints** for the full method and ablations with
MoT, state prediction, or visual patch tokens. The weights-only reconstruction
path supports the legacy CLS-only baseline, not the full architecture.

```bash
python eval_libero.py \
  --checkpoint checkpoints/full_all_data/lewm_step_100000_object.ckpt \
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
python aggregate_all4_results.py \
  --ckpt-dir checkpoints/full_all_data \
  --checkpoint-stem lewm_step_100000_object
```

For a split run, omit `TRAIN_SPLIT=1`; checkpoint selection requires its original
TensorBoard validation logs. To evaluate a particular checkpoint without those
logs, use `eval_libero.py` directly. The aggregation command names the exact
checkpoint so results from different candidates are not combined. With 50
rollouts for every task, the pooled rate equals the mean of the four suite rates.

## Local checks

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -B -m unittest discover -s tests -v
bash -n scripts/preprocess_all4.sh scripts/train_eval.sh
git diff --check
```

Tests cover action codecs, attention visibility, MoT routing, visual token and
projector contracts, and a synthetic HDF5 → model → loss/backward → generation
→ object-checkpoint roundtrip. Synthetic tests use tiny frozen encoder fixtures;
they do not download pretrained weights or run a LIBERO simulator. Training
callback tests additionally require `requirements.txt`.

## Citation

Bibliographic metadata will be added after publication. Until then, cite the
paper by its title, *Autoregressive World Action Model with a Joint-Embedding
Predictive Architecture* (Youwei Li, Yifei Yang, Yue Wang); this is a placeholder,
not a published venue or DOI claim.

## Acknowledgements and license

This work builds on LeWorldModel/LeJEPA, stable-pretraining,
[DINOv2](https://github.com/facebookresearch/dinov2), T5, FAST, and LIBERO.
The inherited MIT license and copyright notice are preserved in [LICENSE](LICENSE).
Datasets, pretrained backbones, and external dependencies retain their own licenses.
