"""Link the four processed LIBERO suites into one flat training directory.

Files keep their original basenames because LiberoDataset sorts filenames before
assigning task IDs and splitting demonstrations. Existing matching links are
reused; conflicting names or destinations are errors. No HDF5 data is copied.

Usage:
    python -m scripts.data.make_flat_dataset
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from jewam.paths import REPO_ROOT

SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def make_flat_dataset(processed_root: Path, output: Path) -> int:
    processed_root = processed_root.resolve()
    output = output.resolve()
    sources: dict[str, Path] = {}
    for suite in SUITES:
        suite_dir = processed_root / suite
        if not suite_dir.is_dir():
            raise FileNotFoundError(f"Missing processed suite: {suite_dir}")
        if output == suite_dir.resolve() or suite_dir.resolve() in output.parents:
            raise ValueError(f"Flat output cannot be inside a source suite: {output}")
        files = sorted([*suite_dir.glob("*.h5"), *suite_dir.glob("*.hdf5")])
        if len(files) != 10:
            raise ValueError(f"Expected 10 HDF5 tasks in {suite_dir}, found {len(files)}")
        for source in files:
            if not source.is_file():
                raise FileNotFoundError(f"Missing HDF5 file or broken link: {source}")
            if source.name in sources:
                raise ValueError(
                    f"Duplicate task basename {source.name}: "
                    f"{sources[source.name]} and {source}"
                )
            sources[source.name] = source

    # Validate every destination before creating anything.
    if output.exists() and not output.is_dir():
        raise NotADirectoryError(output)
    for name, source in sources.items():
        destination = output / name
        if (destination.exists() or destination.is_symlink()) and (
            not destination.is_symlink() or destination.resolve() != source.resolve()
        ):
            raise FileExistsError(f"Refusing to overwrite: {destination}")
    if output.is_dir():
        extras = {
            path.name for path in [*output.glob("*.h5"), *output.glob("*.hdf5")]
        } - sources.keys()
        if extras:
            raise ValueError(f"Unexpected HDF5 files in flat output: {sorted(extras)}")

    output.mkdir(parents=True, exist_ok=True)
    for name, source in sorted(sources.items()):
        destination = output / name
        if not destination.is_symlink():
            destination.symlink_to(os.path.relpath(source, start=output))
    return len(sources)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(
            os.environ.get("PROCESSED_ROOT", REPO_ROOT / "data/libero_processed")
        ),
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Flat output directory; defaults to <processed-root>/all4_flat.",
    )
    args = parser.parse_args()
    output = args.output or args.processed_root / "all4_flat"
    count = make_flat_dataset(args.processed_root, output)
    print(f"Ready: {count} task links in {output.resolve()}")


if __name__ == "__main__":
    main()
