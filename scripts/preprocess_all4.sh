#!/usr/bin/env bash
# Preprocess all 40 LIBERO tasks with a shared FAST tokenizer or scalar bins.
# Raw data is read-only. Existing outputs with the same codec are skipped.
set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    cat <<'HELP'
Usage: [ENV=VALUE ...] bash scripts/preprocess_all4.sh

Environment overrides:
  RAW_ROOT=data/libero_raw      Read-only root with four libero_<suite>/ folders.
  OUT_ROOT=data/libero_processed
  TOKENIZER=data/fast_tokenizer Shared tokenizer from scripts.data.fit_tokenizer_all4.
  CHUNK_SIZE=20 STRIDE=1 MAX_TOKENS=80 PARALLEL=1
  ACTION_CODEC=fast             Or worldvla_bins (MAX_TOKENS defaults to 140).
  NUM_ACTION_BINS=256 ACTION_DIM=7
  PYTHON=python                Interpreter from the active environment.

Paths are resolved from the repository. Outputs may not be inside RAW_ROOT,
including through symbolic links. Existing outputs are skipped after a codec
check; use a separate OUT_ROOT when changing preprocessing settings.
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
RAW_ROOT="${RAW_ROOT:-${REPO_ROOT}/data/libero_raw}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/data/libero_processed}"
TOKENIZER="${TOKENIZER:-${REPO_ROOT}/data/fast_tokenizer}"
CHUNK_SIZE="${CHUNK_SIZE:-20}"
ACTION_DIM="${ACTION_DIM:-7}"
STRIDE="${STRIDE:-1}"
ACTION_CODEC="${ACTION_CODEC:-fast}"
NUM_ACTION_BINS="${NUM_ACTION_BINS:-256}"
if [[ -z "${MAX_TOKENS+x}" ]]; then
    if [[ "$ACTION_CODEC" == "worldvla_bins" ]]; then
        MAX_TOKENS=$((CHUNK_SIZE * ACTION_DIM))
    else
        MAX_TOKENS=80
    fi
fi
PARALLEL="${PARALLEL:-1}"

# xargs spawns child shells via `bash -c` — they do NOT inherit the parent's
# local vars. Export so the per-task run_one() can see them after the fanout.
export CHUNK_SIZE ACTION_DIM STRIDE MAX_TOKENS TOKENIZER ACTION_CODEC NUM_ACTION_BINS PYTHON RAW_ROOT

if [[ "$ACTION_CODEC" != "fast" && "$ACTION_CODEC" != "worldvla_bins" ]]; then
    echo "ERROR: ACTION_CODEC must be fast or worldvla_bins, got: $ACTION_CODEC" >&2
    exit 1
fi
if [[ "$ACTION_CODEC" == "fast" && ! -d "$TOKENIZER" ]]; then
    echo "ERROR: tokenizer dir not found: $TOKENIZER" >&2
    echo "Run python -m scripts.data.fit_tokenizer_all4 first." >&2
    exit 1
fi
if [[ ! -d "$RAW_ROOT" ]]; then
    echo "ERROR: raw root not found: $RAW_ROOT" >&2
    exit 1
fi

# Resolve existing parent links too, so aliases cannot bypass raw-data protection.
check_output_path() {
    "$PYTHON" - "$RAW_ROOT" "$1" <<'PY'
from pathlib import Path
import sys

raw, output = (Path(value).resolve() for value in sys.argv[1:])
if output == raw or raw in output.parents:
    raise SystemExit(f"ERROR: output is inside the read-only raw root: {output}")
PY
}
check_output_path "$OUT_ROOT"

mkdir -p "$OUT_ROOT"

SUITES=("libero_spatial" "libero_object" "libero_goal" "libero_10")

check_existing_codec() {
    local out="$1" expected="$2"
    local actual
    if ! actual=$("$PYTHON" -c 'import h5py, sys
path = sys.argv[1]
with h5py.File(path, "r") as f:
    value = f.attrs.get("action_codec_type", None)
if isinstance(value, bytes):
    value = value.decode("utf-8")
print(value if value is not None else "fast")' "$out"); then
        echo "ERROR: failed to read action_codec_type from existing file: $out" >&2
        return 1
    fi
    if [[ "$actual" != "$expected" ]]; then
        echo "ERROR: $out exists with action_codec_type=$actual, but ACTION_CODEC=$expected." >&2
        echo "Use a codec-specific OUT_ROOT or remove/regenerate the stale file." >&2
        return 1
    fi
}

run_one() {
    local raw="$1" out_suite_dir="$2"
    local base
    base=$(basename "$raw" .hdf5)
    base="${base%_demo}"
    local out="${out_suite_dir}/${base}.h5"
    check_output_path "$out" || return 1
    if [[ -f "$out" ]]; then
        check_existing_codec "$out" "$ACTION_CODEC" || return 1
        echo "[skip] $out exists (action_codec=$ACTION_CODEC)"
        return 0
    fi
    echo "[run] $raw -> $out"
    local codec_args=(--action-codec "$ACTION_CODEC" --num-action-bins "$NUM_ACTION_BINS")
    if [[ "$ACTION_CODEC" == "fast" ]]; then
        codec_args+=(--load-tokenizer "$TOKENIZER")
    fi
    "$PYTHON" -m scripts.data.preprocess_libero \
        --input "$raw" \
        --output "$out" \
        --chunk-size "$CHUNK_SIZE" \
        --stride "$STRIDE" \
        --image-key agentview_rgb \
        --hand-image-key eye_in_hand_rgb \
        --max-action-tokens "$MAX_TOKENS" \
        "${codec_args[@]}"
}

# Build the full task list once, then optionally parallelize.
declare -a JOBS=()
for suite in "${SUITES[@]}"; do
    suite_dir="${RAW_ROOT}/${suite}"
    out_suite_dir="${OUT_ROOT}/${suite}"
    check_output_path "$out_suite_dir"
    mkdir -p "$out_suite_dir"
    if [[ ! -d "$suite_dir" ]]; then
        echo "WARN: suite dir missing: $suite_dir" >&2
        continue
    fi
    for raw in "$suite_dir"/*.hdf5; do
        [[ -e "$raw" ]] || continue
        JOBS+=("$raw|$out_suite_dir")
    done
done

echo "[preprocess_all4] ${#JOBS[@]} tasks to process (PARALLEL=$PARALLEL)"
if [[ "${#JOBS[@]}" -eq 0 ]]; then
    echo "ERROR: no raw .hdf5 tasks found under $RAW_ROOT" >&2
    exit 1
fi

if [[ "$PARALLEL" -le 1 ]]; then
    for entry in "${JOBS[@]}"; do
        raw="${entry%%|*}"; out_suite_dir="${entry##*|}"
        run_one "$raw" "$out_suite_dir"
    done
else
    # Simple N-way fan-out via xargs.
    printf '%s\0' "${JOBS[@]}" | xargs -0 -P "$PARALLEL" -I {} bash -c '
        set -euo pipefail
        entry="$1"
        raw="${entry%%|*}"
        out_suite_dir="${entry##*|}"
        '"$(declare -f check_output_path)"'
        '"$(declare -f check_existing_codec)"'
        '"$(declare -f run_one)"'
        run_one "$raw" "$out_suite_dir"
    ' _ {}
fi

echo "[preprocess_all4] all done"
