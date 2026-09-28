"""Resolve packaged configs separately from local experiment assets."""

from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE_ROOT.parent
_BUNDLED_CONFIG_DIR = _PACKAGE_ROOT / "config" / "train"
_IS_SOURCE_CHECKOUT = (
    not (_BUNDLED_CONFIG_DIR / "jewam.yaml").is_file()
    and (_SOURCE_ROOT / "pyproject.toml").is_file()
    and (_SOURCE_ROOT / "config" / "train" / "jewam.yaml").is_file()
    and (_SOURCE_ROOT / "scripts" / "train_eval.sh").is_file()
)

# Editable/source installs retain repository-relative assets. Wheel installs
# use the current workspace for assets, never the site-packages directory.
REPO_ROOT = _SOURCE_ROOT if _IS_SOURCE_CHECKOUT else Path.cwd()
CONFIG_DIR = (
    _SOURCE_ROOT / "config" / "train"
    if _IS_SOURCE_CHECKOUT
    else _BUNDLED_CONFIG_DIR
)
