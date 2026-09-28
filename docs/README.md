# Documentation

[Project overview](../README.md)

Install the dependency groups described in the [installation guide](installation.md),
then run commands from the repository root in an activated Python 3.10 environment.

## Guides

| Guide | Contents |
|---|---|
| [Installation](installation.md) | Dependency groups, CUDA, and LIBERO |
| [Data and pretrained assets](data.md) | FAST fitting, preprocessing, and the demonstration split |
| [Training and ablations](training.md) | Full model, all-data and split runs, and paper ablations |
| [Evaluation](evaluation.md) | Object checkpoints, rollouts, and result aggregation |
| [Development](development.md) | Local checks and contribution guidelines |

## Source layout

| Path | Contents |
|---|---|
| [`jewam/models/`](../jewam/models/) | Policy, multimodal transformer, and visual backbone |
| [`jewam/actions/`](../jewam/actions/) | Action codecs, FAST decoding, and denormalization |
| [`jewam/data/`](../jewam/data/) | HDF5 dataset, sampling, and shared preprocessing |
| [`jewam/training/`](../jewam/training/) | Training loop, joint losses, and callbacks |
| [`jewam/evaluation/`](../jewam/evaluation/) | Closed-loop LIBERO evaluation |
| [`config/train/`](../config/train/) | Hydra model, training, and dataset configurations |
| [`scripts/`](../scripts/) | Data preparation, experiment runners, and diagnostics |
| [`tests/`](../tests/) | CPU regression and synthetic pipeline tests |
| [`pyproject.toml`](../pyproject.toml) | Package metadata, core dependencies, `train` / `eval` / `dev` extras, and lint settings |
| [`environment.yml`](../environment.yml) | Python 3.10 conda environment with an editable training installation |

For dataset, tokenizer, pretrained model, and checkpoint directories, see the
[local artifact layout](data.md).

## Commands

Use the following entry points; append `--help` for options. The guides above
provide the required data paths and full experiment commands.

| Task | Command |
|---|---|
| Train | `python -m jewam.training.train --config-name=jewam` |
| Evaluate on LIBERO | `python -m jewam.evaluation.libero` |
| Fit the shared FAST tokenizer | `python -m scripts.data.fit_tokenizer_all4` |
| Preprocess one task | `python -m scripts.data.preprocess_libero` |
| Link the processed tasks | `python -m scripts.data.make_flat_dataset` |
| Select checkpoints by validation CE | `python -m scripts.eval.pick_best_ckpt` |
| Aggregate rollout results | `python -m scripts.eval.aggregate_all4_results` |
| Check FAST encoding and decoding | `python -m scripts.diagnostics.check_fast_roundtrip` |
| Check sampler balance | `python -m scripts.diagnostics.verify_sampler_balance` |
| Preprocess all four suites | `bash scripts/preprocess_all4.sh` |
| Train and evaluate all four suites | `bash scripts/train_eval.sh` |
