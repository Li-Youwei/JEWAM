"""Checkpoint discovery must accept current and archived run naming conventions."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HAS_TENSORBOARD = importlib.util.find_spec("tensorboard") is not None
if HAS_TENSORBOARD:
    from tensorboard.compat.proto.event_pb2 import Event
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    from scripts.eval.pick_best_ckpt import find_ckpt_for_step, find_event_files

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(HAS_TENSORBOARD, "training dependencies are required")
class CheckpointSelectionTest(unittest.TestCase):
    def test_step_discovery_accepts_current_and_archived_prefixes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            current = root / "jewam_step_8000_object.ckpt"
            archived = root / "archived_model_step_4000_object.ckpt"
            for path in (current, archived, root / "jewam_latest_object.ckpt"):
                path.touch()
            self.assertEqual(find_ckpt_for_step(root, 8000), current)
            self.assertEqual(find_ckpt_for_step(root, 4000), archived)
            self.assertEqual(find_ckpt_for_step(root, 4100), archived)
            self.assertEqual(find_ckpt_for_step(root, 7999), current)

    def test_discovery_ignores_non_step_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("jewam_latest_object.ckpt", "jewam_step_x_object.ckpt"):
                (root / name).touch()
            self.assertIsNone(find_ckpt_for_step(root, 100))
            self.assertEqual(find_event_files(root), [])

    def test_tensorboard_discovery_accepts_arbitrary_logger_names(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = []
            for logger in ("jewam", "archived_logger"):
                event = root / "tb_logs" / logger / "version_0" / "events.out.tfevents.1"
                event.parent.mkdir(parents=True)
                event.touch()
                expected.append(event)
            unrelated = root / "events.out.tfevents.unrelated"
            unrelated.touch()
            self.assertEqual(find_event_files(root), sorted(expected))

    def test_cli_ranks_actual_events_with_current_and_archived_run_names(self) -> None:
        for prefix, logger in (("jewam", "jewam"), ("archived_model", "archived_logger")):
            with self.subTest(prefix=prefix), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                writer = EventFileWriter(str(root / "tb_logs" / logger / "version_0"))
                for step, value in ((0, 0.01), (4000, 0.5), (8000, 0.25)):
                    writer.add_event(Event(
                        step=step,
                        summary=Summary(value=[Summary.Value(
                            tag="validate/ce_loss_taskbal", simple_value=value,
                        )]),
                    ))
                writer.close()
                for step in (4000, 8000):
                    (root / f"{prefix}_step_{step}_object.ckpt").touch()
                result = subprocess.run(
                    [sys.executable, "-B", "-m", "scripts.eval.pick_best_ckpt",
                     "--ckpt-dir", str(root), "--top-k", "2"],
                    cwd=ROOT, capture_output=True, text=True, check=True,
                )
                selected = json.loads(result.stdout)
                self.assertEqual(selected["top_1"], str(root / f"{prefix}_step_8000_object.ckpt"))
                self.assertEqual([entry["step"] for entry in selected["all"]], [8000, 4000])


if __name__ == "__main__":
    unittest.main()
