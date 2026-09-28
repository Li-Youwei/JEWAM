# JEWAM

**JEWAM (Joint-Embedding World Action Model)** implements the method in
*Autoregressive World Action Model with a Joint-Embedding Predictive Architecture*.
The policy learns robot manipulation from the 40 tasks in four LIBERO suites,
without robot-data pretraining. It predicts FAST action tokens from two RGB
views, language, and proprioception. Future-state prediction supervises training;
rollouts use the action branch only.

This repository includes the full method and the components needed for its
ablations. Pretrained policy checkpoints, demonstrations, and manuscript files
are not included.

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
runs; see the [training protocols](docs/training.md).

## Layout

```text
jewam/models/        Policy, MoT/Transformer, and frozen vision backbone
jewam/actions/       Action codecs, FAST decoding, and denormalization
jewam/data/          Dataset, balanced sampling, and shared preprocessing
jewam/training/      Training runner, joint losses, and callbacks
jewam/evaluation/    Closed-loop LIBERO evaluation
config/train/        Hydra model, training, and dataset configurations
docs/                Installation, data, experiments, and contributor guides
scripts/data/        Preprocessing, FAST fitting, and dataset preparation
scripts/eval/        Checkpoint selection and rollout result aggregation
scripts/diagnostics/ Action codec and sampling checks
scripts/*.sh         End-to-end preprocessing and training/evaluation runners
tests/               CPU regression tests and synthetic pipeline smoke tests
pyproject.toml       Package metadata, dependency groups, and lint settings
environment.yml     Python 3.10 training environment
```

See the [documentation](docs/README.md) for setup, training, evaluation, and
development guides.

## Quick start

Use Python 3.10 and run commands from the repository root. Full training and
LIBERO rollouts require Linux/CUDA; CPU checks can run on macOS. Install the
package and training dependencies with `python -m pip install -e '.[train]'`;
see the installation guide for environment setup and other dependency groups.

1. [Install the environment](docs/installation.md).
2. [Prepare LIBERO data, frozen encoders, and the shared FAST tokenizer](docs/data.md).
3. [Train the full model or paper ablations](docs/training.md).
4. [Evaluate object checkpoints on all four suites](docs/evaluation.md).
5. [Run local checks and follow the contribution contracts](docs/development.md).

After installing dependencies and preparing assets and data, the default 9:1
split experiment is:

```bash
CUDA_VISIBLE_DEVICES=0 ARM=full SEED=3072 RUN_NAME=full_split \
  bash scripts/train_eval.sh
```

For the all-demonstration main-result protocol, set `TRAIN_SPLIT=1` and use a new
`RUN_NAME`; see [training protocols](docs/training.md).

## Citation

Bibliographic metadata will be added after publication. Until then, cite the
paper by its title, *Autoregressive World Action Model with a Joint-Embedding
Predictive Architecture* (Youwei Li, Yifei Yang, Yue Wang); this is a placeholder,
not a published venue or DOI claim.

## Acknowledgements and license

JEWAM uses stable-pretraining for the training loop,
[DINOv2](https://github.com/facebookresearch/dinov2) and T5 for frozen features,
FAST for action tokenization, and LIBERO for demonstrations and evaluation.
The code is distributed under the [MIT license](LICENSE). The copyright notice
for inherited SIGReg and basic neural-network utilities is retained alongside
the JEWAM contributors' notice. Datasets, pretrained backbones, and external
dependencies retain their own licenses.
