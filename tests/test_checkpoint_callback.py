"""Object checkpoints for fixed-budget retraining and validation-based runs."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from torch import nn


class ObjectCheckpointCallbackTest(unittest.TestCase):
    def setUp(self):
        try:
            from utils import ModelObjectCallBack
        except ModuleNotFoundError as exc:
            if exc.name == "lightning":
                self.skipTest(
                    "Checkpoint callback tests require the lightning training dependency"
                )
            raise
        self.callback_class = ModelObjectCallBack
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = nn.Linear(3, 2)
        self.module = SimpleNamespace(model=self.model)
        self.trainer = SimpleNamespace(
            is_global_zero=True, global_step=100_000, callback_metrics={}
        )

    def test_all_demonstration_run_saves_final_object_without_validation(self):
        callback = self.callback_class(
            self.root,
            filename="jewam",
            step_interval=4000,
            save_final_on_train_end=True,
        )
        callback.on_train_end(self.trainer, self.module)
        path = self.root / "jewam_step_100000_object.ckpt"
        self.assertTrue(path.is_file())
        restored = torch.load(path, weights_only=False)
        inputs = torch.tensor([[1.0, 2.0, 3.0]])
        torch.testing.assert_close(restored(inputs), self.model(inputs), rtol=0, atol=0)

    def test_final_save_is_opt_in_and_global_zero_only(self):
        for enabled, is_global_zero in [(False, True), (True, False)]:
            with self.subTest(enabled=enabled, is_global_zero=is_global_zero):
                callback = self.callback_class(
                    self.root,
                    filename="jewam",
                    step_interval=4000,
                    save_final_on_train_end=enabled,
                )
                self.trainer.is_global_zero = is_global_zero
                callback.on_train_end(self.trainer, self.module)
                self.assertEqual(list(self.root.iterdir()), [])

    def test_validation_ranking_and_latest_link_remain_unchanged(self):
        callback = self.callback_class(
            self.root, filename="jewam", step_interval=4000, top_k=1
        )
        for step, score in [(4000, 0.5), (8000, 0.8)]:
            self.trainer.global_step = step
            self.trainer.callback_metrics = {
                "validate/ce_loss_taskbal": torch.tensor(score)
            }
            callback.on_validation_end(self.trainer, self.module)
        self.trainer.global_step = 100_000
        callback.on_train_end(self.trainer, self.module)
        best = self.root / "jewam_step_4000_object.ckpt"
        self.assertTrue(best.is_file())
        self.assertFalse((self.root / "jewam_step_8000_object.ckpt").exists())
        self.assertFalse((self.root / "jewam_step_100000_object.ckpt").exists())
        self.assertEqual((self.root / "jewam_latest_object.ckpt").resolve(), best.resolve())


if __name__ == "__main__":
    unittest.main()
