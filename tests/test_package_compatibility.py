"""Object checkpoints keep their legacy imports while new saves use the package."""

from __future__ import annotations

import ast
import importlib
import pickle
import pickletools
import subprocess
import sys
import textwrap
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
LEGACY_CLASSES = {
    "jepa": ("jewam.models.policy", ("JEPA",)),
    "module": (
        "jewam.models.transformer",
        (
            "SIGReg",
            "FeedForward",
            "Attention",
            "Block",
            "ModalityLayerNorm",
            "MoTAttention",
            "MoTBlock",
            "Transformer",
            "MLP",
            "ARPredictor",
        ),
    ),
    "vision_backbone": ("jewam.models.vision", ("HFVisionBackbone",)),
}


def _checkpoint_globals(path: Path) -> set[tuple[str, str]]:
    with zipfile.ZipFile(path) as archive:
        data_file = next(name for name in archive.namelist() if name.endswith("/data.pkl"))
        data = archive.read(data_file)
    return {
        tuple(argument.split(" ", 1))
        for opcode, argument, _ in pickletools.genops(data)
        if opcode.name == "GLOBAL"
    }


def _checkpoint_payload():
    from jewam.models.policy import JEPA
    from jewam.models.transformer import MLP, ARPredictor, SIGReg, Transformer
    from jewam.models.vision import HFVisionBackbone

    torch.manual_seed(3072)
    # The synthetic forward starts at encoded visual features. The backbone
    # wrapper is still nested in the object, so its old pickle path is exercised.
    encoder = nn.Identity()
    encoder.config = SimpleNamespace(hidden_size=16)
    model = JEPA(
        encoder=HFVisionBackbone(encoder),
        predictor=ARPredictor(
            embed_dim=16,
            depth=1,
            heads=2,
            dim_head=8,
            mlp_dim=32,
            max_action_tokens=4,
            max_lang_tokens=3,
            dropout=0.0,
            emb_dropout=0.0,
            use_state_prediction=True,
            visual_pool_grid=2,
            state_pred_visual_tokens=True,
            state_prediction_arch="mot",
        ),
        projector=MLP(16, 32, 16, norm_fn=nn.BatchNorm1d),
        patch_projector=MLP(16, 32, 16, norm_fn=nn.LayerNorm),
        visual_pool_grid=2,
        freeze_encoder=True,
    ).eval()
    inputs = (
        torch.randn(2, 5, 16),
        torch.randn(2, 5, 16),
        torch.randn(2, 9),
        torch.randn(2, 3, 16),
        torch.tensor([3, 2]),
        torch.tensor([[1, 2, 3, 1026], [4, 5, 1026, 1026]]),
        torch.tensor([3, 2]),
    )
    with torch.no_grad():
        expected = model.predict(*inputs)
        generated = model.predict_actions(*inputs[:5], max_len=4)
    return {
        "model": model,
        "inputs": inputs,
        "expected": expected,
        "generated": generated,
        # Cover the shared-transformer ablation classes and training regularizer
        # as well as every class nested in the full MoT policy.
        "shared_transformer": Transformer(16, 16, 16, 1, 2, 8, 32),
        "sigreg": SIGReg(knots=5, num_proj=8),
    }


class PackageCompatibilityTest(unittest.TestCase):
    def test_legacy_pickle_globals_resolve_to_canonical_classes(self):
        for legacy_name, (canonical_name, names) in LEGACY_CLASSES.items():
            legacy = importlib.import_module(legacy_name)
            canonical = importlib.import_module(canonical_name)
            for name in names:
                with self.subTest(module=legacy_name, name=name):
                    cls = getattr(canonical, name)
                    self.assertIs(getattr(legacy, name), cls)
                    self.assertEqual(cls.__module__, canonical_name)
                    # Protocol 0 GLOBAL is exactly the module/name lookup used
                    # by old object checkpoints, independent of new save paths.
                    old_global = f"c{legacy_name}\n{name}\n.".encode("ascii")
                    self.assertIs(pickle.loads(old_global), cls)

    def _assert_model_globals(self, path: Path, *, legacy: bool):
        globals_ = _checkpoint_globals(path)
        for legacy_name, (canonical_name, names) in LEGACY_CLASSES.items():
            expected_module = legacy_name if legacy else canonical_name
            forbidden_module = canonical_name if legacy else legacy_name
            for name in names:
                self.assertIn((expected_module, name), globals_)
                self.assertNotIn((forbidden_module, name), globals_)

    def _load_in_fresh_process(self, source: Path, resaved: Path, *, legacy: bool):
        code = textwrap.dedent(
            """
            import sys
            import torch
            from jewam.models.policy import JEPA
            from jewam.models.transformer import ARPredictor, MoTBlock, Transformer
            from jewam.models.vision import HFVisionBackbone

            roots = {'jepa', 'module', 'vision_backbone'}
            assert roots.isdisjoint(sys.modules), roots.intersection(sys.modules)
            payload = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
            model = payload['model'].eval()
            assert type(model) is JEPA
            assert type(model.predictor) is ARPredictor
            assert type(model.predictor.blocks[0]) is MoTBlock
            assert type(model.encoder) is HFVisionBackbone
            assert type(payload['shared_transformer']) is Transformer
            with torch.no_grad():
                actual = model.predict(*payload['inputs'])
                generated = model.predict_actions(*payload['inputs'][:5], max_len=4)
            for value, expected in zip(actual, payload['expected']):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            for value, expected in zip(generated, payload['generated']):
                torch.testing.assert_close(value, expected, rtol=0, atol=0)
            if sys.argv[3] == 'legacy':
                assert roots.issubset(sys.modules)
            else:
                assert roots.isdisjoint(sys.modules), roots.intersection(sys.modules)
            torch.save(payload, sys.argv[2], pickle_protocol=2)
            """
        )
        result = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                code,
                str(source),
                str(resaved),
                "legacy" if legacy else "canonical",
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_nested_legacy_checkpoint_loads_and_resaves_canonical_paths(self):
        payload = _checkpoint_payload()
        with TemporaryDirectory() as temp:
            legacy_path = Path(temp) / "legacy_object.ckpt"
            migrated_path = Path(temp) / "migrated_object.ckpt"
            with ExitStack() as stack:
                for legacy_name, (canonical_name, names) in LEGACY_CLASSES.items():
                    importlib.import_module(legacy_name)
                    canonical = importlib.import_module(canonical_name)
                    for name in names:
                        stack.enter_context(
                            patch.object(getattr(canonical, name), "__module__", legacy_name)
                        )
                torch.save(payload, legacy_path, pickle_protocol=2)
            self._assert_model_globals(legacy_path, legacy=True)
            self._load_in_fresh_process(legacy_path, migrated_path, legacy=True)
            self._assert_model_globals(migrated_path, legacy=False)

    def test_new_checkpoint_does_not_import_legacy_shims(self):
        with TemporaryDirectory() as temp:
            source = Path(temp) / "canonical_object.ckpt"
            resaved = Path(temp) / "resaved_object.ckpt"
            torch.save(_checkpoint_payload(), source, pickle_protocol=2)
            self._assert_model_globals(source, legacy=False)
            self._load_in_fresh_process(source, resaved, legacy=False)

    def test_core_package_does_not_depend_on_scripts_or_root_modules(self):
        forbidden = {
            "scripts",
            "jepa",
            "module",
            "vision_backbone",
            "action_codec",
            "fast_utils",
            "libero_dataset",
            "utils",
            "train",
            "eval_libero",
        }
        package = ROOT / "jewam"
        self.assertTrue(package.is_dir())
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    modules = [node.module or ""]
                else:
                    continue
                for module in modules:
                    with self.subTest(path=path.relative_to(ROOT), line=node.lineno):
                        self.assertNotIn(module.split(".", 1)[0], forbidden)


if __name__ == "__main__":
    unittest.main()
