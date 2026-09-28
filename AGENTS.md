# Contributor guide

JEWAM is maintained on `main`, the only development baseline.
Use current code and `config/train/jewam.yaml` as the source of truth. Do not
restore implementations from old branches or worktrees.

## Repository layout

- Put implementations in `jewam/`: models, actions, data, training, and
  evaluation. Shared preprocessing belongs in `jewam/data/preprocessing.py`.
  Core code must import `jewam.*`, never `scripts` or the root compatibility
  modules; scripts handle arguments and workflow orchestration.
- Keep `jepa.py`, `module.py`, and `vision_backbone.py` as explicit compatibility
  re-exports, and `train.py` / `eval_libero.py` as thin CLI entry points only.
  Do not put shared functions back in the root wrappers.
- Run canonical CLIs with `python -m jewam.training.train` and
  `python -m jewam.evaluation.libero`. The documented experiment workflow uses
  an editable install and runs commands from the repository root.
- Keep Hydra configs in `config/train/`; resolve them through
  `jewam.paths.CONFIG_DIR`. Packaging maps these files to `jewam/config/train/`
  without duplicating the source configs. `REPO_ROOT` locates experiment assets:
  the checkout root for source/editable installs, the working directory for wheels.
- Put data preparation tools in `scripts/data/`, checkpoint/result tools in
  `scripts/eval/`, and standalone checks in `scripts/diagnostics/`. Run them
  from the repository root with `python -m scripts.<group>.<tool>`.
- Keep the end-to-end shell runners at `scripts/preprocess_all4.sh` and
  `scripts/train_eval.sh`; run these from the source checkout. Keep tests in
  `tests/` and preserve diagnostic tools and configs used to validate experiments.
- Keep `__init__.py` markers for the regular Python packages in `jewam/`,
  `scripts/`, and `tests/`. Register new distributable packages explicitly in
  `pyproject.toml`; exclude tests and local experiment artifacts from the wheel.
- Keep README focused on the method, results, and getting started. Put detailed
  setup and experiment instructions in `docs/`; see [the guide](docs/README.md).
  Public documentation should describe current usage, not internal cleanup or
  migration history.

## Architecture contracts

- Frozen DINOv2-base and T5-small; dual RGB views and 9D proprioception.
- Each view contributes CLS plus 16 pooled patches. Independent CLS/patch
  projectors are 768 → 2048 → 192, with BatchNorm/LayerNorm respectively.
- Six MoT layers, 16 heads, head dimension 64, FFN 2048, dropout 0.2.
- FAST vocabulary 1024, action budget 80, chunk horizon 20, action dimension 7.
- Future supervision predicts 17 visual tokens per view and 9D proprioception.
  Targets retain projector gradients. Visual loss equally weights CLS MSE and
  patch MSE; SIGReg receives current/future CLS streams only.
- Full loss is CE + 1.0 SP + 0.1 SIGReg, label smoothing 0.1; gripper aux is off.
- Keep attention masks, generation, anchor-relative actions, quaternion order,
  both raw gripper finger positions, and closed-loop control semantics intact.
- Keep the root module/class names in `jepa.py`, `module.py`, and
  `vision_backbone.py` as aliases to the canonical model classes: existing
  pickled object checkpoints depend on them. Do not remove these compatibility
  modules; new checkpoints use the canonical `jewam.*` class paths.
- Shared Transformer, identity projection, and uniform-bin action code paths
  are needed for paper ablations. Other compatibility paths are not the full
  model's recommended configuration.

## Data and experiments

- Treat the configured raw LIBERO directory as read-only. Write derived data
  to a separate directory. The preprocessing runner checks resolved paths.
- Preprocessing computes task normalization and fits the shared FAST tokenizer
  before the demo-level train/validation split; do not silently change this.
- Sorted task filenames define task IDs and split ordering. Preserve filenames
  and the seed to reproduce existing splits.
- `scripts/train_eval.sh` defaults to `ARM=full`, seed 3072, batch 128, 100000
  steps, 2000 warmup steps, and `TRAIN_SPLIT=0.9`.
- Split runs rank object checkpoints by task-balanced validation CE and evaluate
  the top three with 50 rollouts per task. There is no light-eval selection.
- `TRAIN_SPLIT=1` disables validation and evaluates the final fixed-budget
  object checkpoint. Only this case enables the final-save callback.
- Keep `trainer.devices=1` for BatchNorm/SIGReg models; no SyncBatchNorm is added.
- Use `base.yaml` for shared config defaults and `jewam.yaml` for the full
  model. New checkpoints use `jewam_*_object.ckpt`; selection also accepts
  existing checkpoints with other prefixes.
- Do not change defaults or loss/data semantics merely to simplify code.

## Development

Use Python 3.10 and an activated environment. See
[installation](docs/installation.md) for dependency groups.
`pyproject.toml` is the single dependency source: core dependencies plus `train`,
`eval`, and `dev` extras. Do not maintain duplicate requirements files.
`environment.yml` selects Python 3.10 and installs the editable `train` extra.
`stable-pretraining==0.1.6` preserves the Manager API used by this training code.
Dependencies are release compatibility constraints, not the historical server
lockfile. No server access is needed for repository maintenance.

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

Core tests and the synthetic pipeline do not need downloaded weights or LIBERO.
Callback tests need the training dependencies. Full training and simulator
rollouts require Linux/CUDA, real data, and pretrained assets; do not claim they
ran when only CPU checks ran. Use `JEWAM_HOME` to override the checkpoint root.
Keep manuscripts, datasets, fitted tokenizers, weights, logs, and machine-local
credentials out of Git. Preserve the inherited MIT notice.
Finder layout metadata (`.DS_Store`), Python caches, and build outputs stay local
and ignored. When changing packaging, verify editable and wheel imports, bundled
Hydra configs, CLI entry points, and the three checkpoint compatibility modules.
