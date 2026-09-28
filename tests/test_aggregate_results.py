"""Reject incomplete rollout results before reporting a four-suite score."""

from __future__ import annotations

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from aggregate_all4_results import SUITES, main


class AggregateResultsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.stem = "jewam_step_100000_object"
        for suite in SUITES:
            self.write_log(suite)

    def log_path(self, suite):
        return self.root / f"eval_{self.stem}_{suite}.log"

    def write_log(self, suite, task_ids=range(10), successes=1, episodes=2):
        self.log_path(suite).write_text(
            "".join(
                f"  Task {task}: {successes}/{episodes} (50.0%) — task_{task}\n"
                for task in task_ids
            )
        )

    def run_aggregate(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        args = [
            "aggregate_all4_results.py", "--ckpt-dir", str(self.root),
            "--checkpoint-stem", self.stem,
        ]
        with patch("sys.argv", args), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            status = main()
        return status, stdout.getvalue(), stderr.getvalue()

    def assert_incomplete(self):
        status, stdout, stderr = self.run_aggregate()
        self.assertNotEqual(status, 0)
        self.assertNotIn("**overall**", stdout)
        self.assertIn("incomplete four-suite evaluation", stderr)

    def test_complete_run_allows_nondefault_episode_count(self):
        status, stdout, stderr = self.run_aggregate()
        self.assertEqual(status, 0, stderr)
        self.assertIn("**40 / 80**", stdout)
        self.assertRegex(stdout, r"\*\*\s*50\.0%\*\*")

    def test_missing_suite_is_rejected(self):
        self.log_path(SUITES[0]).unlink()
        self.assert_incomplete()

    def test_empty_suite_is_rejected(self):
        self.log_path(SUITES[0]).write_text("")
        self.assert_incomplete()

    def test_truncated_suite_is_rejected(self):
        self.write_log(SUITES[0], task_ids=range(9))
        self.assert_incomplete()

    def test_wrong_task_ids_are_rejected(self):
        self.write_log(SUITES[0], task_ids=range(1, 11))
        self.assert_incomplete()

    def test_zero_episode_task_is_rejected(self):
        self.write_log(SUITES[0], task_ids=range(9))
        with self.log_path(SUITES[0]).open("a") as stream:
            stream.write("  Task 9: 0/0 (0.0%) — task_9\n")
        self.assert_incomplete()

    def test_impossible_success_count_is_rejected(self):
        self.write_log(SUITES[0], successes=3, episodes=2)
        self.assert_incomplete()

    def test_repeated_summary_does_not_double_count(self):
        path = self.log_path(SUITES[0])
        path.write_text(path.read_text() * 2)
        status, stdout, stderr = self.run_aggregate()
        self.assertEqual(status, 0, stderr)
        self.assertIn("**40 / 80**", stdout)


if __name__ == "__main__":
    unittest.main()
