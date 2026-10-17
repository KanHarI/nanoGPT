"""Recovery decisions must never duplicate healthy training or stop it early."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.finite_field.watchdog import CONDITIONS, check_once, decide, inspect_runs


class WatchdogTests(unittest.TestCase):
    def test_healthy_training_is_left_running_even_after_retries(self):
        snapshot = dict(runs=[dict(last_progress_unix=999)], all_complete=False,
                        comparison_complete=False)
        self.assertEqual(decide(snapshot, True, 2, 1000)[0], "monitor")

    def test_complete_reports_wait_for_service_exit_and_comparison(self):
        snapshot = dict(runs=[dict(last_progress_unix=999)], all_complete=True,
                        comparison_complete=True)
        self.assertEqual(decide(snapshot, True, 0, 1000)[0], "monitor")
        self.assertEqual(decide(snapshot, False, 0, 1000)[0], "shutdown_complete")
        snapshot["comparison_complete"] = False
        self.assertEqual(decide(snapshot, False, 0, 1000)[0], "resume")

    def test_recovery_is_bounded_and_stall_detection_allows_startup(self):
        snapshot = dict(runs=[dict(last_progress_unix=100)], all_complete=False,
                        comparison_complete=False)
        self.assertEqual(decide(snapshot, True, 0, 2000)[0], "resume")
        self.assertEqual(decide(snapshot, True, 2, 2000)[0], "shutdown_failed")
        self.assertEqual(decide(snapshot, False, 0, 2000)[0], "resume")
        self.assertEqual(decide(snapshot, True, 1, 2000,
                                previous=dict(last_restart_unix=1900))[0], "monitor")
        snapshot["runs"] = []
        self.assertEqual(decide(snapshot, True, 0, 2000,
                                previous=dict(first_check_unix=100))[0], "resume")

    def test_apply_leaves_healthy_training_untouched(self):
        snapshot = dict(runs=[dict(last_progress_unix=10**12)], all_complete=False,
                        comparison_complete=False)
        with tempfile.TemporaryDirectory() as temporary, \
                patch('experiments.finite_field.watchdog.inspect_runs', return_value=snapshot), \
                patch('experiments.finite_field.watchdog.command',
                      return_value=SimpleNamespace(stdout='active')) as execute:
            result = check_once(Path(temporary), apply=True)
            self.assertEqual(result['action'], 'monitor')
            self.assertTrue(all(call.args[0] == 'systemctl' for call in execute.call_args_list))

    def test_recovery_cancels_shutdown_and_records_retry_before_start(self):
        snapshot = dict(runs=[], all_complete=False, comparison_complete=False)
        with tempfile.TemporaryDirectory() as temporary, \
                patch('experiments.finite_field.watchdog.inspect_runs', return_value=snapshot), \
                patch('experiments.finite_field.watchdog.command',
                      return_value=SimpleNamespace(stdout='inactive')) as execute:
            result = check_once(Path(temporary), apply=True)
            self.assertEqual(result['retries'], 1)
            calls = [call.args for call in execute.call_args_list]
            cancel = ('sudo', 'shutdown', '-c')
            start = ('sudo', 'systemctl', 'start', 'finite-field-runner.service')
            self.assertLess(calls.index(cancel), calls.index(start))
            self.assertEqual(json.loads((Path(temporary)/'watchdog.json').read_text())['retries'], 1)

    def test_incomplete_or_mismatched_final_is_not_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition in CONDITIONS:
                path = root / f"{condition}-seed0"
                path.mkdir()
                config = dict(condition=condition, seed=0, steps=10)
                (path / "config.json").write_text(json.dumps(config))
                (path / "final.json").write_text(json.dumps(dict(step=10, config=config)))
                (path / "metrics.jsonl").write_text(
                    json.dumps(dict(kind="train", step=10, gamma=12)) + '\n{"kind":')
            snapshot = inspect_runs(root)
            self.assertTrue(snapshot["all_complete"])
            self.assertFalse(snapshot["comparison_complete"])
            self.assertEqual(snapshot["runs"][0]["step"], 10)
            (root / "hard-seed0" / "final.json").write_text(json.dumps(
                dict(step=9, config=dict(condition="hard", seed=0, steps=10))))
            self.assertFalse(inspect_runs(root)["all_complete"])


if __name__ == "__main__":
    unittest.main()
