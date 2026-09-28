"""Offline smoke tests across preprocessing, HDF5 loading, and the model.

The synthetic token ids and tiny frozen encoders replace downloaded FAST/T5/
DINO assets. Action extraction, normalization, dataset loading, JEPA/MoT,
training loss, generation, and object serialization use production code.
"""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from torch import nn
from torch.utils.data import DataLoader

from jewam.data.libero import LiberoDataset
from jewam.data.preprocessing import (
    compute_action_stats,
    extract_chunks,
    normalize_actions,
    save_hdf5,
)
from jewam.models.policy import JEPA
from jewam.models.transformer import MLP, ARPredictor, SIGReg


class TinyVisionEncoder(nn.Module):
    """Image-dependent, pickleable stand-in for a frozen visual backbone."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 24)

    def forward(self, images, interpolate_pos_encoding=True):
        patches = nn.functional.adaptive_avg_pool2d(images, (4, 4))
        patches = self.projection(patches.flatten(2).transpose(1, 2))
        return SimpleNamespace(
            last_hidden_state=torch.cat([patches.mean(1, keepdim=True), patches], dim=1)
        )


class TinyLanguageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 12)
        self.requires_grad_(False)

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class TinyTokenizer:
    def __call__(self, text, *, max_length, **kwargs):
        ids = torch.zeros(1, max_length, dtype=torch.long)
        ids[0, :3] = torch.tensor([2, 3, 1])
        return {"input_ids": ids, "attention_mask": (ids != 0).long()}


class PipelineSmokeTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3072)
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.raw = self.root / "raw.hdf5"
        processed = self.root / "processed"
        processed.mkdir()
        self.horizon = 20
        self.steps = 23
        rng = np.random.default_rng(3072)
        t = np.arange(self.steps, dtype=np.float64)
        self.positions = np.stack([0.001 * t**2, 0.002 * t, -0.001 * t], axis=1)
        self.rotvec = np.stack([np.zeros_like(t), np.zeros_like(t), 0.01 * t], axis=1)
        self.images = rng.integers(0, 256, (self.steps, 8, 8, 3), dtype=np.uint8)
        self.gripper = np.where(t % 2 == 0, -1.0, 1.0)
        with h5py.File(self.raw, "w") as f:
            for demo_id in range(2):
                demo = f.create_group(f"data/demo_{demo_id}")
                obs = demo.create_group("obs")
                actions = np.zeros((self.steps, 7), dtype=np.float32)
                actions[:, 6] = self.gripper
                demo["actions"] = actions
                robot_states = np.zeros((self.steps, 9), dtype=np.float64)
                robot_states[:, 5:9] = Rotation.from_rotvec(self.rotvec).as_quat()
                demo["robot_states"] = robot_states
                obs["ee_pos"] = self.positions
                obs["ee_ori"] = self.rotvec
                obs["gripper_states"] = np.tile([0.02, -0.02], (self.steps, 1))
                obs["agentview_rgb"] = self.images
                obs["eye_in_hand_rgb"] = 255 - self.images
        with redirect_stdout(StringIO()), h5py.File(self.raw, "r") as f:
            self.samples = extract_chunks(
                f, ["demo_0", "demo_1"], "agentview_rgb", "eye_in_hand_rgb", 20, 1
            )
            self.physical_actions = np.stack(self.samples["continuous_actions"])
            low, high = compute_action_stats(self.samples["continuous_actions"])
            self.samples["continuous_actions"] = [
                normalize_actions(chunk, low, high) for chunk in self.physical_actions
            ]
            # Different lengths exercise the actual padding and shifted-EOS path.
            tokens = [np.arange(3 + i % 2, dtype=np.int32) + 7 for i in range(6)]
            save_hdf5(
                str(processed / "synthetic_task.h5"),
                self.samples,
                tokens,
                low,
                high,
                chunk_size=20,
                chunk_stride=1,
                image_key="agentview_rgb",
                source_file=str(self.raw),
                num_demos=2,
                language_instruction="move the block",
                language_source="synthetic",
            )
        with patch(
            "jewam.data.libero.T5Tokenizer.from_pretrained", return_value=TinyTokenizer()
        ):
            self.dataset = LiberoDataset(
                str(processed),
                max_action_tokens=6,
                max_lang_tokens=5,
                img_size=16,
                use_language=True,
                use_state_prediction=True,
                action_codec_type="fast",
            )
        self.addCleanup(self.dataset.__del__)
        self.batch = next(iter(DataLoader(self.dataset, batch_size=2, shuffle=False)))

    def _model(self):
        return JEPA(
            encoder=TinyVisionEncoder(),
            predictor=ARPredictor(
                embed_dim=16,
                depth=1,
                heads=2,
                dim_head=8,
                mlp_dim=32,
                max_action_tokens=6,
                max_lang_tokens=5,
                proprio_dim=9,
                dropout=0.0,
                emb_dropout=0.0,
                use_state_prediction=True,
                state_head_norm_type="batch",
                visual_pool_grid=4,
                state_pred_visual_tokens=True,
                state_prediction_arch="mot",
            ),
            projector=MLP(24, 32, 16, norm_fn=nn.BatchNorm1d),
            patch_projector=MLP(24, 32, 16, norm_fn=nn.LayerNorm),
            lang_encoder=TinyLanguageEncoder(),
            lang_proj=nn.Linear(12, 16),
            visual_pool_grid=4,
            freeze_encoder=True,
        )

    def _encode(self, model):
        return model.encode(
            self.batch["pixels_agent"],
            self.batch["pixels_hand"],
            self.batch["lang_input_ids"],
            self.batch["lang_attention_mask"],
        )

    def _predict(self, model):
        agent, hand, language, lengths = self._encode(model)
        return model.predict(
            agent,
            hand,
            self.batch["proprio"],
            language,
            lengths,
            self.batch["action_tokens"],
            self.batch["action_lengths"],
        )

    def test_anchor_relative_preprocessing_and_dataset_contract(self):
        self.assertEqual(len(self.dataset), 6)
        self.assertEqual(self.samples["chunk_idx"], [0, 1, 2, 0, 1, 2])
        np.testing.assert_allclose(
            self.physical_actions[0, :, :3],
            self.positions[1:21] - self.positions[0],
            rtol=1e-6,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            self.physical_actions[0, :, 3:6], self.rotvec[1:21], atol=1e-7
        )
        np.testing.assert_array_equal(self.physical_actions[0, :, 6], self.gripper[:20])
        np.testing.assert_array_equal(
            self.samples["image_agent_future"][0], self.images[20]
        )
        np.testing.assert_allclose(self.samples["proprio"][0][-2:], [0.02, -0.02])
        torch.testing.assert_close(
            self.batch["proprio_future"][0, :3],
            torch.tensor(self.positions[20]).float(),
        )
        self.assertEqual(tuple(self.batch["pixels_agent"].shape), (2, 3, 16, 16))
        self.assertEqual(tuple(self.batch["gripper_seq"].shape), (2, 20))
        self.assertEqual(self.batch["action_lengths"].tolist(), [3, 4])
        torch.testing.assert_close(
            self.dataset.get_sampler_weights(),
            torch.full((6,), 1 / 6, dtype=torch.double),
        )

    def test_forward_generation_and_object_checkpoint_roundtrip(self):
        model = self._model().eval()
        with torch.no_grad():
            before = self._predict(model)
        self.assertEqual(
            [tuple(x.shape) for x in before],
            [(2, 7, 1026), (2, 17, 16), (2, 17, 16), (2, 9)],
        )
        path = self.root / "smoke_object.ckpt"
        torch.save(model, path)
        restored = torch.load(path, map_location="cpu", weights_only=False).eval()
        with torch.no_grad():
            after = self._predict(restored)
        for expected, actual in zip(before, after):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        agent, hand, language, lengths = self._encode(restored)
        # Future-state heads must be absent from the inference execution path.
        with patch.object(
            restored.predictor.state_pred_head_ag,
            "forward",
            side_effect=AssertionError("SP head used during inference"),
        ):
            tokens, token_lengths = restored.predict_actions(
                agent,
                hand,
                self.batch["proprio"],
                language,
                lengths,
                max_len=4,
            )
        self.assertEqual(tokens.shape[0], 2)
        self.assertTrue((token_lengths <= 4).all())
        for row, length in zip(tokens, token_lengths):
            self.assertTrue(((row[:length] >= 0) & (row[:length] < 1024)).all())

    def test_production_joint_loss_backpropagates_with_frozen_encoders(self):
        try:
            from jewam.training.train import jewam_forward
        except ModuleNotFoundError as exc:
            if exc.name in {"hydra", "omegaconf"}:
                self.skipTest(f"Training loss smoke requires hydra-core: {exc}")
            raise
        model = self._model().train()
        self.assertFalse(model.encoder.training)
        self.assertFalse(model.lang_encoder.training)
        logged = {}
        runner = SimpleNamespace(
            model=model,
            sigreg=SIGReg(knots=5, num_proj=8),
            log_dict=lambda values, **kwargs: logged.update(values),
        )
        cfg = {
            "loss": {"pred_weight": 1.0, "sigreg_weight": 0.1},
            "visual_tokens": {"patch_sp": True, "patch_sp_weight": 1.0},
            "label_smoothing": 0.1,
        }
        output = jewam_forward(runner, self.batch, "fit", cfg)
        self.assertTrue(torch.isfinite(output["loss"]))
        torch.testing.assert_close(
            output["loss"],
            output["ce_loss"] + output["pred_loss"] + 0.1 * output["sigreg_loss"],
        )
        self.assertIn("fit/pred_loss_ag_patch", logged)
        output["loss"].backward()
        for component in [
            model.projector,
            model.patch_projector,
            model.lang_proj,
            model.predictor,
        ]:
            gradients = [p.grad for p in component.parameters() if p.requires_grad]
            self.assertTrue(gradients)
            self.assertTrue(
                all(g is not None and torch.isfinite(g).all() for g in gradients)
            )
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(all(p.grad is None for p in model.lang_encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
