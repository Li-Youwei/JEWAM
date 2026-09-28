"""Tool relocation must preserve module entry points and repository asset paths."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from jewam.data import preprocessing
from scripts.data import fit_tokenizer_all4, make_flat_dataset
from scripts.diagnostics import check_fast_roundtrip

ROOT = Path(__file__).resolve().parents[1]


class _ArgumentsParsed(Exception):
    """Stop a CLI after real argument parsing, before it touches data or models."""


class ScriptLayoutTest(unittest.TestCase):
    def test_module_entrypoints_show_help_without_assets(self) -> None:
        modules = (
            "scripts.data.fit_tokenizer_all4",
            "scripts.data.preprocess_libero",
            "scripts.data.make_flat_dataset",
            "scripts.eval.pick_best_ckpt",
            "scripts.eval.aggregate_all4_results",
            "scripts.diagnostics.check_fast_roundtrip",
            "scripts.diagnostics.verify_sampler_balance",
        )
        for module in modules:
            with self.subTest(module=module):
                if module.endswith("pick_best_ckpt") and importlib.util.find_spec(
                    "tensorboard"
                ) is None:
                    self.skipTest("checkpoint selection requires tensorboard")
                result = subprocess.run(
                    [sys.executable, "-B", "-m", module, "--help"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("usage:", result.stdout.lower())

    def test_default_asset_paths_still_resolve_from_repository_root(self) -> None:
        original_parse_args = argparse.ArgumentParser.parse_args
        cases = (
            (fit_tokenizer_all4.main, {
                "raw_root": ROOT / "data/libero_raw",
                "tokenizer_out": ROOT / "data/fast_tokenizer",
                "audit_out": ROOT / "data/libero_processed/audit_token_length.json",
            }),
            (make_flat_dataset.main, {
                "processed_root": ROOT / "data/libero_processed",
            }),
            (check_fast_roundtrip.main, {
                "processed_dir": ROOT / "data/libero_processed/libero_spatial",
                "tokenizer": ROOT / "data/fast_tokenizer",
            }),
        )
        for main, expected in cases:
            with self.subTest(module=main.__module__):
                parsed = []

                def capture_arguments(parser):
                    parsed.append(original_parse_args(parser, []))
                    raise _ArgumentsParsed

                with (
                    patch.dict(os.environ, {}, clear=True),
                    patch.object(
                        argparse.ArgumentParser, "parse_args", capture_arguments
                    ),
                    self.assertRaises(_ArgumentsParsed),
                ):
                    main()
                for name, path in expected.items():
                    self.assertEqual(getattr(parsed[0], name), path)

    def test_tokenizer_processor_fallback_still_uses_root_data_directory(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "data/fast_tokenizer/processing_action_tokenizer.py"
            source.parent.mkdir(parents=True)
            source.write_text("class UniversalActionProcessor: pass\n")
            output = root / "saved_tokenizer"
            output.mkdir()
            with patch.object(preprocessing, "REPO_ROOT", root):
                preprocessing._patch_saved_tokenizer(output)
            self.assertEqual(
                (output / source.name).read_text(), source.read_text()
            )
            config = json.loads((output / "processor_config.json").read_text())
            self.assertEqual(
                config["auto_map"]["AutoProcessor"],
                "processing_action_tokenizer.UniversalActionProcessor",
            )


if __name__ == "__main__":
    unittest.main()
