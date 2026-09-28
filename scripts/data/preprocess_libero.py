"""Convert LIBERO demonstrations into chunk-level samples using shared preprocessing.

Run from the repository root with ``python -m scripts.data.preprocess_libero``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

from jewam.actions.codec import FAST_CODEC, WORLDVLA_BINS_CODEC
from jewam.data.preprocessing import (
    compute_action_stats,
    encode_action_chunks,
    extract_chunks,
    extract_language_instruction,
    inspect_hdf5,
    load_demo_keys,
    normalize_actions,
    print_statistics,
    save_hdf5,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess LIBERO HDF5 demos into chunk-level action-tokenized samples."
    )
    parser.add_argument(
        "--input", required=True,
        help="Path to input LIBERO HDF5 file (one task).",
    )
    parser.add_argument(
        "--output", required=True,
        help="Path for the output HDF5 file.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=20,
        help="Action chunk length H in raw env steps (default: 20 = 1s at 20Hz).",
    )
    parser.add_argument(
        "--stride", type=int, default=1,
        help="Sliding window stride in raw steps (default: 1). stride=1 is the "
             "standard VLA practice — consecutive chunks share H-1 actions but "
             "each gets a fresh observation anchor, maximizing data coverage. "
             "Using stride=H would reduce per-demo sample count by ~H and is "
             "strongly discouraged.",
    )
    parser.add_argument(
        "--image-key", default="agentview_rgb",
        help="Agentview image key in the HDF5 (default: agentview_rgb).",
    )
    parser.add_argument(
        "--hand-image-key", default="eye_in_hand_rgb",
        help="Eye-in-hand image key in the HDF5 (default: eye_in_hand_rgb).",
    )
    parser.add_argument(
        "--max-action-tokens", type=int, default=None,
        help="Assert that no token sequence exceeds this length. If None, skip assertion.",
    )
    parser.add_argument(
        "--action-codec",
        choices=[FAST_CODEC, WORLDVLA_BINS_CODEC],
        default=FAST_CODEC,
        help="Action token codec: FAST DCT+BPE or WorldVLA-style scalar bins.",
    )
    parser.add_argument(
        "--num-action-bins",
        type=int,
        default=256,
        help="Number of scalar bins for --action-codec=worldvla_bins.",
    )
    parser.add_argument(
        "--fit-tokenizer", action="store_true",
        help="Train a LIBERO-specific FAST BPE tokenizer instead of using the universal one.",
    )
    parser.add_argument(
        "--save-tokenizer", default=None,
        help="Directory to save the fitted FAST tokenizer.",
    )
    parser.add_argument(
        "--load-tokenizer", default=None,
        help="Directory to load a previously fitted FAST tokenizer.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Step 1: Load and inspect
    print(f"Opening {args.input} ...")
    with h5py.File(args.input, "r") as f:
        demo_keys = load_demo_keys(f)
        if not demo_keys:
            sys.exit("ERROR: No demo_* groups found under data/")
        inspect_hdf5(f, args.image_key, args.hand_image_key, demo_keys)

        # Extract language instruction
        language_instruction, language_source = extract_language_instruction(
            f,
            args.input,
        )
        print(
            f"  Language instruction: '{language_instruction}' "
            f"(source={language_source})"
        )

        # Step 2+3: Extract anchor-relative sliding-window chunks in PHYSICAL units.
        # Stats are computed from the chunks themselves (not raw HDF5 actions) so
        # that the 1/99 percentile reflects the actual distribution the model sees.
        samples = extract_chunks(
            f, demo_keys, args.image_key, args.hand_image_key,
            args.chunk_size, stride=args.stride,
        )

    if not samples["continuous_actions"]:
        sys.exit("ERROR: No chunks extracted. Check trajectory lengths vs chunk size.")

    # Compute per-dim percentile stats from the anchor-relative chunks.
    action_low, action_high = compute_action_stats(samples["continuous_actions"])

    # Normalize chunks in place to [-1, 1] using the fitted bounds.
    samples["continuous_actions"] = [
        normalize_actions(chunk, action_low, action_high)
        for chunk in samples["continuous_actions"]
    ]

    # Step 4: action tokenization (on normalized anchor-relative chunks).
    all_action_chunks = np.stack(samples["continuous_actions"], axis=0)  # (N, H, 7)
    tokens_list, _tokenizer, action_codec = encode_action_chunks(
        all_action_chunks,
        action_codec=args.action_codec,
        num_action_bins=args.num_action_bins,
        fit=args.fit_tokenizer,
        save_tokenizer_path=args.save_tokenizer,
        load_tokenizer_path=args.load_tokenizer,
    )

    # Token length assertion (CRITICAL — see AGENTS.md)
    # Do NOT silently truncate; raise an error so max_action_tokens can be increased.
    max_observed = max(len(t) for t in tokens_list)
    print(f"  Max observed token length: {max_observed}")
    if args.max_action_tokens is not None and max_observed > args.max_action_tokens:
        sys.exit(
            f"ERROR: max observed action token length ({max_observed}) exceeds "
            f"max_action_tokens ({args.max_action_tokens}). Increase max_action_tokens "
            f"in config and rerun."
        )

    # Step 5: Save output HDF5
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_hdf5(
        args.output, samples, tokens_list, action_low, action_high,
        chunk_size=args.chunk_size,
        chunk_stride=args.stride,
        image_key=args.image_key,
        source_file=args.input,
        num_demos=len(demo_keys),
        language_instruction=language_instruction,
        language_source=language_source,
        action_codec=action_codec,
        save_tokenizer_path=args.save_tokenizer,
        load_tokenizer_path=args.load_tokenizer,
    )

    # Step 6: Print statistics
    print_statistics(
        samples,
        tokens_list,
        action_low,
        action_high,
        action_codec_name=action_codec.name,
    )

    print("Done!")


if __name__ == "__main__":
    main()
