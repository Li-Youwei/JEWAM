#!/usr/bin/env bash
# Train and evaluate JEWAM on all four LIBERO suites.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: [ENV=VALUE ...] bash scripts/train_eval.sh

Defaults reproduce the full model with a 9:1 demonstration split. Set
TRAIN_SPLIT=1 to retrain on all demonstrations and evaluate the final checkpoint.
Activate the Python environment first, or set PYTHON to its Python executable.

Environment overrides:
  ARM=full                     Full 192-D, patch-SP, MoT model; legacy ARM names work.
  SEED=3072 MAX_STEPS=100000    Training seed and optimization-step budget.
  TRAIN_SPLIT=0.9              Training demonstration fraction, in (0, 1].
  BATCH_SIZE=128 GRAD_ACCUM_STEPS=1 WARMUP_STEPS=2000 VAL_INTERVAL=4000
  PRED_WEIGHT=1.0 SIGREG_WEIGHT=0.1
  FLAT_DIR=data/libero_processed/all4_flat
  PROCESSED_ROOT=data/libero_processed TOKENIZER=data/fast_tokenizer
  VISION_ENCODER=pretrained/dinov2-base CKPT_ROOT=checkpoints
  RUN_NAME=<ARM>_seed<SEED>     Use distinct names for ablations and full-data runs.
  EVAL_ONLY=false              Evaluate an existing run without training.
  CKPT_SELECT_TOP_K=3 CKPT_SAVE_TOP_K=-1 FINAL_EVAL_EPISODES=50
  FINAL_EVAL_MAX_STEPS=openvla EVAL_NUM_STEPS_WAIT=10
  ACTION_CODEC=fast            Or worldvla_bins, for the tokenization ablation.
  PYTHON=python                Interpreter from the active environment.

Default paths are rooted at the repository; relative overrides are also resolved
from the repository. GPU visibility is controlled by CUDA_VISIBLE_DEVICES.
HELP
    exit 0
fi
if [[ $# -gt 0 ]]; then
    echo "ERROR: unexpected argument: $1 (use --help)" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON="${PYTHON:-python}"
command -v "$PYTHON" >/dev/null || { echo "ERROR: Python not found: $PYTHON; activate your environment or set PYTHON." >&2; exit 1; }

ARM="${ARM:-full}"
PROJECTOR_TYPE_DEFAULT=mlp
PROJECTOR_DEPTH_DEFAULT=2
EMBED_DIM_DEFAULT=192
SP_TARGET_SPACE_DEFAULT=projected
PATCH_PROJECTOR_NORM_DEFAULT=layer
STATE_HEAD_NORM_DEFAULT=""
case "$ARM" in
  all4_dinov2_frozen)
    PRED=0
    SIGREG=0
    NORM=layer
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=0
    PATCH_SP_DEFAULT=false
    ;;
  all4_dinov2_frozen_visual17)
    PRED=0
    SIGREG=0
    NORM=layer
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=false
    ;;
  all4_dinov2_frozen_visual17_mot)
    PRED=0
    SIGREG=0
    NORM=layer
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=false
    ;;
  all4_sigreg_dinov2_frozen_visual17_mot)
    PRED=0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=0
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_sep_proj)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_patch_sp)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=shared
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=true
    ;;
  full|all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=true
    ;;
  all4_sp_sigreg_dinov2_frozen_visual17_patch_sp_mot_dino192_mlp6)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=true
    PROJECTOR_TYPE_DEFAULT=mlp
    PROJECTOR_DEPTH_DEFAULT=6
    EMBED_DIM_DEFAULT=192
    SP_TARGET_SPACE_DEFAULT=projected
    PATCH_PROJECTOR_NORM_DEFAULT=layer
    STATE_HEAD_NORM_DEFAULT=batch
    ;;
  all4_sp_sigreg_dinov2_frozen_mot)
    PRED=1.0
    SIGREG=0.1
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=0
    PATCH_SP_DEFAULT=false
    ;;
  all4_sp_dinov2_frozen_visual17_patch_sp_mot_dino768_mlp)
    PRED=1.0
    SIGREG=0
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=true
    PROJECTOR_TYPE_DEFAULT=mlp
    EMBED_DIM_DEFAULT=768
    SP_TARGET_SPACE_DEFAULT=projected
    PATCH_PROJECTOR_NORM_DEFAULT=layer
    STATE_HEAD_NORM_DEFAULT=batch
    ;;
  all4_sp_dinov2_frozen_visual17_patch_sp_mot_dino768_identity)
    PRED=1.0
    SIGREG=0
    NORM=batch
    STATE_ARCH_DEFAULT=mot
    POOL_GRID_DEFAULT=4
    PATCH_SP_DEFAULT=true
    PROJECTOR_TYPE_DEFAULT=identity
    EMBED_DIM_DEFAULT=768
    SP_TARGET_SPACE_DEFAULT=backbone
    PATCH_PROJECTOR_NORM_DEFAULT=layer
    STATE_HEAD_NORM_DEFAULT=batch
    ;;
  *)
    echo "Unknown ARM: $ARM" >&2
    echo "Expected a supported all4 DINOv2 arm; got: $ARM" >&2
    exit 1
    ;;
esac

PRED="${PRED_WEIGHT:-$PRED}"
SIGREG="${SIGREG_WEIGHT:-$SIGREG}"
SEED="${SEED:-3072}"
MAX_STEPS="${MAX_STEPS:-100000}"
TRAIN_SPLIT="${TRAIN_SPLIT:-0.9}"
FULL_DATA=$("$PYTHON" - "$TRAIN_SPLIT" <<'PY'
import sys

split = float(sys.argv[1])
if not 0 < split <= 1:
    raise SystemExit("ERROR: TRAIN_SPLIT must be in (0, 1].")
print("true" if split == 1 else "false")
PY
)
EVAL_ONLY="${EVAL_ONLY:-false}"
VAL_INTERVAL="${VAL_INTERVAL:-4000}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"
STATE_ARCH="${STATE_ARCH:-$STATE_ARCH_DEFAULT}"
POOL_GRID="${POOL_GRID:-$POOL_GRID_DEFAULT}"
PATCH_SP="${PATCH_SP:-$PATCH_SP_DEFAULT}"
PATCH_SP_WEIGHT="${PATCH_SP_WEIGHT:-1.0}"
SP_TARGET_SPACE="${SP_TARGET_SPACE:-$SP_TARGET_SPACE_DEFAULT}"
PROJECTOR_TYPE="${PROJECTOR_TYPE:-$PROJECTOR_TYPE_DEFAULT}"
PROJECTOR_DEPTH="${PROJECTOR_DEPTH:-$PROJECTOR_DEPTH_DEFAULT}"
EMBED_DIM="${EMBED_DIM:-$EMBED_DIM_DEFAULT}"
PATCH_PROJECTOR_NORM="${PATCH_PROJECTOR_NORM:-$PATCH_PROJECTOR_NORM_DEFAULT}"
STATE_HEAD_NORM="${STATE_HEAD_NORM:-${STATE_HEAD_NORM_DEFAULT:-$NORM}}"
CKPT_SELECT_TOP_K="${CKPT_SELECT_TOP_K:-3}"
CKPT_SAVE_TOP_K="${CKPT_SAVE_TOP_K:--1}"
FINAL_EVAL_EPISODES="${FINAL_EVAL_EPISODES:-50}"
FINAL_EVAL_MAX_STEPS="${FINAL_EVAL_MAX_STEPS:-openvla}"
EVAL_NUM_STEPS_WAIT="${EVAL_NUM_STEPS_WAIT:-10}"
ACTION_CODEC="${ACTION_CODEC:-fast}"
NUM_ACTION_BINS="${NUM_ACTION_BINS:-256}"
ACTION_DIM="${ACTION_DIM:-7}"
if (( BATCH_SIZE < 1 || GRAD_ACCUM_STEPS < 1 )); then
    echo "ERROR: BATCH_SIZE and GRAD_ACCUM_STEPS must be positive integers" >&2
    exit 1
fi
if (( CKPT_SAVE_TOP_K == 0 || CKPT_SAVE_TOP_K < -1 )); then
    echo "ERROR: CKPT_SAVE_TOP_K must be -1 (keep all) or a positive integer" >&2
    exit 1
fi
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS))
if [[ -z "${MAX_ACTION_TOKENS+x}" ]]; then
    if [[ "$ACTION_CODEC" == "worldvla_bins" ]]; then
        MAX_ACTION_TOKENS=$((20 * ACTION_DIM))
    else
        MAX_ACTION_TOKENS=80
    fi
fi

PROCESSED_ROOT="${PROCESSED_ROOT:-${REPO_ROOT}/data/libero_processed}"
FLAT_DIR="${FLAT_DIR:-${PROCESSED_ROOT}/all4_flat}"
TOKENIZER="${TOKENIZER:-${REPO_ROOT}/data/fast_tokenizer}"
VISION_ENCODER="${VISION_ENCODER:-${REPO_ROOT}/pretrained/dinov2-base}"
CKPT_ROOT="${CKPT_ROOT:-${REPO_ROOT}/checkpoints}"
RUN_NAME="${RUN_NAME:-${ARM}_seed${SEED}}"
CKPT_DIR="${CKPT_ROOT}/${RUN_NAME}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export JEWAM_HOME="$CKPT_DIR"

resolve_eval_max_steps() {
    local suite="$1"
    local configured="$2"
    if [[ "$configured" != "openvla" ]]; then
        echo "$configured"
        return
    fi
    case "$suite" in
      libero_spatial) echo 220 ;;
      libero_object) echo 280 ;;
      libero_goal) echo 300 ;;
      libero_10) echo 520 ;;
      libero_90) echo 400 ;;
      *)
        echo "ERROR: no OpenVLA horizon for suite: $suite" >&2
        return 1
        ;;
    esac
}

echo "=========================================================="
echo "[JEWAM] ARM=$ARM STATE_ARCH=$STATE_ARCH SEED=$SEED MAX_STEPS=$MAX_STEPS"
echo "[JEWAM] EVAL_ONLY=$EVAL_ONLY TRAIN_SPLIT=$TRAIN_SPLIT"
echo "[JEWAM] PRED=$PRED SIGREG=$SIGREG NORM=$NORM"
echo "[JEWAM] PROJECTOR_TYPE=$PROJECTOR_TYPE PROJECTOR_DEPTH=$PROJECTOR_DEPTH EMBED_DIM=$EMBED_DIM STATE_HEAD_NORM=$STATE_HEAD_NORM PATCH_PROJECTOR_NORM=$PATCH_PROJECTOR_NORM"
echo "[JEWAM] POOL_GRID=$POOL_GRID PATCH_SP=$PATCH_SP PATCH_SP_WEIGHT=$PATCH_SP_WEIGHT SP_TARGET_SPACE=$SP_TARGET_SPACE"
echo "[JEWAM] BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS EFFECTIVE_BATCH_SIZE=$EFFECTIVE_BATCH_SIZE"
echo "[JEWAM] FLAT_DIR=$FLAT_DIR"
echo "[JEWAM] TOKENIZER=$TOKENIZER"
echo "[JEWAM] PROCESSED_ROOT=$PROCESSED_ROOT"
echo "[JEWAM] VISION_ENCODER=$VISION_ENCODER"
echo "[JEWAM] CKPT_DIR=$CKPT_DIR"
echo "[JEWAM] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}"
echo "[JEWAM] CKPT_SAVE_TOP_K=$CKPT_SAVE_TOP_K CKPT_SELECT_TOP_K=$CKPT_SELECT_TOP_K FINAL_EVAL_EPISODES=$FINAL_EVAL_EPISODES FINAL_EVAL_MAX_STEPS=$FINAL_EVAL_MAX_STEPS EVAL_NUM_STEPS_WAIT=$EVAL_NUM_STEPS_WAIT"
echo "[JEWAM] ACTION_CODEC=$ACTION_CODEC NUM_ACTION_BINS=$NUM_ACTION_BINS MAX_ACTION_TOKENS=$MAX_ACTION_TOKENS"
echo "=========================================================="

[[ -d "$FLAT_DIR" ]] || { echo "ERROR: FLAT_DIR missing: $FLAT_DIR" >&2; exit 1; }
if [[ "$ACTION_CODEC" != "fast" && "$ACTION_CODEC" != "worldvla_bins" ]]; then
    echo "ERROR: ACTION_CODEC must be fast or worldvla_bins, got: $ACTION_CODEC" >&2
    exit 1
fi
if [[ "$ACTION_CODEC" == "fast" ]]; then
    [[ -d "$TOKENIZER" ]] || { echo "ERROR: TOKENIZER missing: $TOKENIZER" >&2; exit 1; }
fi
[[ -d "$VISION_ENCODER" ]] || {
    echo "ERROR: VISION_ENCODER missing: $VISION_ENCODER" >&2
    echo "Download/copy a HuggingFace vision model there first." >&2
    exit 1
}
N_FLAT=$(find -H "$FLAT_DIR" -maxdepth 1 \( -name "*.h5" -o -name "*.hdf5" \) | wc -l)
if [[ "$N_FLAT" -lt 40 ]]; then
    echo "ERROR: expected 40 HDF5 tasks in $FLAT_DIR, got $N_FLAT" >&2
    exit 1
fi

mkdir -p "$CKPT_DIR"
TRAIN_LOG="${CKPT_DIR}/train.log"

EVAL_CODEC_ARGS=(--action-codec "$ACTION_CODEC" --num-action-bins "$NUM_ACTION_BINS")
if [[ "$ACTION_CODEC" == "fast" ]]; then
    EVAL_CODEC_ARGS+=(--tokenizer "$TOKENIZER")
fi
TRAIN_SPLIT_ARGS=("train_split=$TRAIN_SPLIT")
if [[ "$FULL_DATA" == "true" ]]; then
    TRAIN_SPLIT_ARGS+=("+trainer.limit_val_batches=0")
fi

if [[ "$EVAL_ONLY" == "true" || "$EVAL_ONLY" == "1" ]]; then
    echo "[JEWAM] EVAL_ONLY enabled; skipping training and using existing checkpoints in $CKPT_DIR"
else
    "$PYTHON" train.py --config-name=base \
        data=libero \
        data.dataset.hdf5_dir="$FLAT_DIR" \
        data.dataset.max_action_tokens="$MAX_ACTION_TOKENS" \
        data.dataset.action_dim="$ACTION_DIM" \
        data.dataset.action_codec.type="$ACTION_CODEC" \
        data.dataset.action_codec.num_bins="$NUM_ACTION_BINS" \
        vision_encoder.source=hf \
        vision_encoder.model_name_or_path="$VISION_ENCODER" \
        vision_encoder.freeze=true \
        vision_encoder.local_files_only=true \
        loss.pred_weight="$PRED" \
        loss.sigreg_weight="$SIGREG" \
        wm.embed_dim="$EMBED_DIM" \
        predictor.state_prediction_arch="$STATE_ARCH" \
        predictor.state_head_norm_type="$STATE_HEAD_NORM" \
        projector.type="$PROJECTOR_TYPE" \
        projector.depth="$PROJECTOR_DEPTH" \
        projector.norm_type="$NORM" \
        scheduler.warmup_steps="$WARMUP_STEPS" \
        trainer.devices=1 \
        +trainer.max_steps="$MAX_STEPS" \
        trainer.max_epochs=999 \
        +trainer.val_check_interval="$VAL_INTERVAL" \
        +trainer.check_val_every_n_epoch=null \
        +trainer.accumulate_grad_batches="$GRAD_ACCUM_STEPS" \
        loader.batch_size="$BATCH_SIZE" \
        seed="$SEED" \
        subdir="" \
        output_model_name=jewam \
        +ckpt_top_k="$CKPT_SAVE_TOP_K" \
        +visual_tokens.pool_grid="$POOL_GRID" \
        +visual_tokens.patch_sp="$PATCH_SP" \
        +visual_tokens.patch_sp_weight="$PATCH_SP_WEIGHT" \
        +visual_tokens.patch_projector_norm_type="$PATCH_PROJECTOR_NORM" \
        +visual_tokens.sp_target_space="$SP_TARGET_SPACE" \
        "${TRAIN_SPLIT_ARGS[@]}" \
        2>&1 | tee "$TRAIN_LOG"
fi

CANDIDATE_CKPTS=()
if [[ "$FULL_DATA" == "true" ]]; then
    FINAL_CKPT="${CKPT_DIR}/jewam_step_${MAX_STEPS}_object.ckpt"
    if [[ ! -f "$FINAL_CKPT" ]]; then
        # Earlier runs may use another filename prefix; the step stays exact.
        shopt -s nullglob
        FINAL_MATCHES=("${CKPT_DIR}/"*_step_"${MAX_STEPS}"_object.ckpt)
        shopt -u nullglob
        if [[ "${#FINAL_MATCHES[@]}" -eq 1 ]]; then
            FINAL_CKPT="${FINAL_MATCHES[0]}"
        elif [[ "${#FINAL_MATCHES[@]}" -gt 1 ]]; then
            echo "ERROR: multiple final-step checkpoints found in $CKPT_DIR" >&2
            exit 2
        fi
    fi
    CANDIDATE_CKPTS+=("$FINAL_CKPT")
    echo "[JEWAM] full-data run: evaluating the final step, without validation selection"
else
PICK_JSON="${CKPT_DIR}/ckpt_ce_topk.json"
PICK_OUT=$("$PYTHON" pick_best_ckpt.py --ckpt-dir "$CKPT_DIR" --top-k "$CKPT_SELECT_TOP_K")
echo "[JEWAM] pick_best_ckpt output:"
echo "$PICK_OUT"
echo "$PICK_OUT" > "$PICK_JSON"

while IFS= read -r candidate; do
    CANDIDATE_CKPTS+=("$candidate")
done < <("$PYTHON" - "$PICK_JSON" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path) as f:
    data = json.load(f)
seen = set()
for row in data.get("all", []):
    ckpt = row.get("ckpt")
    if ckpt and ckpt not in seen:
        seen.add(ckpt)
        print(ckpt)
PY
)
fi
if [[ "${#CANDIDATE_CKPTS[@]}" -eq 0 ]]; then
    echo "ERROR: pick_best_ckpt returned no candidate checkpoints" >&2
    exit 2
fi

for candidate_ckpt in "${CANDIDATE_CKPTS[@]}"; do
    if [[ ! -f "$candidate_ckpt" ]]; then
        echo "ERROR: candidate ckpt missing: $candidate_ckpt" >&2
        exit 2
    fi
    candidate_stem="$(basename "$candidate_ckpt" .ckpt)"
    echo "[JEWAM] eval candidate: $candidate_ckpt"
    for suite in libero_spatial libero_object libero_goal libero_10; do
        suite_max_steps="$(resolve_eval_max_steps "$suite" "$FINAL_EVAL_MAX_STEPS")"
        EVAL_LOG="${CKPT_DIR}/eval_${candidate_stem}_${suite}.log"
        PROC_DIR="${PROCESSED_ROOT}/${suite}"
        [[ -d "$PROC_DIR" ]] || { echo "ERROR: processed suite dir missing: $PROC_DIR" >&2; exit 1; }
        echo "[JEWAM] eval $suite episodes_per_task=$FINAL_EVAL_EPISODES max_steps=$suite_max_steps -> $EVAL_LOG"
        "$PYTHON" eval_libero.py \
            --checkpoint "$candidate_ckpt" \
            --processed-dir "$PROC_DIR" \
            --suite "$suite" \
            --num-episodes "$FINAL_EVAL_EPISODES" \
            --max-steps "$suite_max_steps" \
            --num-steps-wait "$EVAL_NUM_STEPS_WAIT" \
            --device cuda \
            --seed "$SEED" \
            "${EVAL_CODEC_ARGS[@]}" \
            2>&1 | tee "$EVAL_LOG"
    done
done

echo "[JEWAM] ALL DONE - see logs under $CKPT_DIR/"
