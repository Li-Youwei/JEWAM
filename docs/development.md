# Development

[Documentation index](README.md) · [Project overview](../README.md)

Run all commands from the repository root in an activated environment.

```bash
python -m pip install -e '.[train,dev]'
ruff check .
python -B -m unittest discover -s tests -v
python -m jewam.training.train --config-name=jewam --cfg job --resolve
python -m jewam.evaluation.libero --help
python -m scripts.data.preprocess_libero --help
python -m scripts.eval.pick_best_ckpt --help
python -m scripts.diagnostics.check_fast_roundtrip --help
bash -n scripts/preprocess_all4.sh scripts/train_eval.sh
git diff --check
```

Tests cover action codecs, attention visibility, MoT routing, visual token and
projector contracts, and a synthetic HDF5 → model → loss/backward → generation
→ object-checkpoint roundtrip. Synthetic tests use tiny frozen encoder fixtures;
they do not download pretrained weights or run a LIBERO simulator. The `train`
extra provides dependencies for training callback tests, the checkpoint-selection
CLI, and the training configuration check. For core CPU checks only, install
`python -m pip install -e '.[dev]'`; see [installation](installation.md).

Full training and simulator rollouts require Linux/CUDA, real data, and
pretrained assets. Passing CPU checks does not establish rollout success or
reproduce the reported paper results. No server access is needed for repository
maintenance.

Follow the [contributor contracts](../AGENTS.md) and [repository layout](README.md)
when making changes. Preserve model import/class names used by object
checkpoints, architecture and attention semantics, loss and data protocols,
checkpoint selection, and closed-loop control. Keep the shared Transformer,
identity projection, and uniform-bin paths required for paper ablations.

Implement shared functionality inside `jewam/`. Core modules must not import
`scripts` or the root compatibility wrappers. In particular, preprocessing
functions belong in `jewam.data.preprocessing`; the scripts only parse arguments
and orchestrate workflows. Preserve the root model aliases needed for
[object checkpoint loading](evaluation.md#checkpoint-compatibility).

Keep manuscript files, datasets, fitted tokenizers, weights, logs, and
machine-local credentials out of Git, and preserve the inherited MIT copyright
notice.
