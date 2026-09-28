# Contributor guide

JEWAM is the public release of the final `codex/bestmodel` implementation.
Use current code and `config/train/jewam.yaml` as the source of truth. Do not
restore implementations from old branches or worktrees.

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
  `vision_backbone.py`: existing pickled object checkpoints depend on them.
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
- Legacy base config and `lewm_*_object.ckpt` filenames are intentional.
- Do not change defaults or loss/data semantics merely to simplify code.

## Development

Use Python 3.10 and an activated environment. See README for dependency groups.
`stable-pretraining==0.1.6` preserves the Manager API used by this training code.
Dependencies are release compatibility constraints, not the historical server
lockfile. No server access is needed for repository maintenance.

```bash
python -m pip install -r requirements-dev.txt
ruff check .
python -B -m unittest discover -s tests -v
python train.py --config-name=jewam --cfg job --resolve
python eval_libero.py --help
bash -n scripts/preprocess_all4.sh scripts/train_eval.sh
git diff --check
```

Core tests and the synthetic pipeline do not need downloaded weights or LIBERO.
Callback tests need the training dependencies. Full training and simulator
rollouts require Linux/CUDA, real data, and pretrained assets; do not claim they
ran when only CPU checks ran. Keep manuscripts, datasets, fitted tokenizers,
weights, logs, and machine-local credentials out of Git. Preserve the inherited
MIT notice.
