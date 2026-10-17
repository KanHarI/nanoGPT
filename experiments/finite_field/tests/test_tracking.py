"""Tracking must preserve training, committed history, and run identity."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

from experiments.finite_field.config import Config
from experiments.finite_field.tracking import Tracking, Tracker, import_history
from experiments.finite_field.train import run


def tiny_config():
    return Config(steps=6, batch_size=8, block_size=16, max_operators=4,
                  n_layer=1, n_head=2, n_embd=16, dropout=0.1,
                  gamma_low=0, gamma_high=2, warmup_steps=1, eval_interval=6,
                  eval_per_field=2, final_per_field=2, baseline_per_field=4,
                  eval_gammas=(0, 2), log_interval=2, threads=1)


def fake_sdk():
    sdk = MagicMock()

    def initialize(**kwargs):
        # SDK initialization is allowed to consume RNG; training still must match.
        torch.rand(10)
        session = MagicMock()
        session.id, session.project = kwargs["id"], kwargs["project"]
        session.entity = kwargs["entity"] or "test-user"
        session.url = f"https://wandb.ai/test-user/{session.project}/runs/{session.id}"
        session.summary = {}
        return session

    sdk.init.side_effect = initialize
    return sdk


class TrackingTests(unittest.TestCase):
    def test_same_optimizer_step_keeps_train_and_eval_and_baselines(self):
        sdk = fake_sdk()
        with tempfile.TemporaryDirectory() as temporary, patch.dict("sys.modules", wandb=sdk):
            with Tracker(Tracking(mode="online"), tiny_config(), temporary) as tracker:
                tracker.log(dict(kind="train", step=2, gamma=1, loss=2.3,
                                 operator_counts={"+": 3}, train_seconds=2, supervised_answers=16))
                tracker.log(dict(kind="eval", step=2, loss=2.1, by_field={"19": {"loss": 2.8}}))
                calls = tracker.run.log.call_args_list
                self.assertEqual(len(calls), 2)
                train, evaluation = [call.args[0] for call in calls]
                self.assertEqual(train["optimizer_step"], evaluation["optimizer_step"])
                self.assertEqual(train["curriculum/gamma"], 1)
                self.assertEqual(train["train/operator_counts/add"], 3)
                self.assertEqual(train["throughput/answers_per_second"], 8)
                self.assertEqual(evaluation["eval/high_gamma/by_field/19/loss"], 2.8)
                self.assertAlmostEqual(evaluation["baseline/uniform_loss"], 2.010950560525004)
                self.assertAlmostEqual(evaluation["baseline/uniform_accuracy"], 0.18193471904772215)

    def test_tracked_resume_matches_untracked_training_and_reuses_identity(self):
        config, sdk = tiny_config(), fake_sdk()
        options = Tracking(mode="online", group="test")
        with tempfile.TemporaryDirectory() as temporary, patch.dict("sys.modules", wandb=sdk):
            full, split = Path(temporary) / "full", Path(temporary) / "split"
            run(config, full)
            sdk.init.assert_not_called()
            run(config, split, stop_after=3, tracking=options)
            identity = json.loads((split / "wandb_run.json").read_text())
            run(config, split, resume=True, tracking=options, final_evaluation=True)
            self.assertEqual(json.loads((split / "wandb_run.json").read_text()), identity)
            self.assertEqual(sdk.init.call_args.kwargs["resume"], "must")
            self.assertEqual(sdk.init.call_args.kwargs["id"], identity["id"])
            a = torch.load(full / "checkpoint.pt", weights_only=False)
            b = torch.load(split / "checkpoint.pt", weights_only=False)
            self.assertEqual(a["sampler"], b["sampler"])
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            # A config mismatch must fail before contacting W&B.
            sdk.init.reset_mock()
            with self.assertRaises(ValueError):
                run(replace(config, steps=8), split, resume=True, tracking=options)
            sdk.init.assert_not_called()

    def test_history_import_excludes_uncommitted_tail_and_marks_partial_budget(self):
        config, sdk = tiny_config(), fake_sdk()
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary) / "run"
            checkpoint_path = run(config, directory, stop_after=3)
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            with (directory / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"kind": "uncommitted"}) + "\n")
            # A report from another checkpoint must not be attached to this run.
            (directory / "final.json").write_text(json.dumps({"step": 6}))
            with patch.dict("sys.modules", wandb=sdk):
                sessions = []
                initialize = sdk.init.side_effect
                def capture(**kwargs):
                    sessions.append(initialize(**kwargs))
                    return sessions[-1]
                sdk.init.side_effect = capture
                import_history(directory, Tracking(mode="online"))
                session = sessions[-1]
                self.assertEqual(session.log.call_count, checkpoint["records"])
                self.assertEqual(session.summary["training/status"], "checkpointed")
                self.assertEqual(session.summary["training/completed_updates"], 3)
                uploaded_names = [call.kwargs["name"] for call in sdk.Artifact.return_value.add_file.call_args_list]
                self.assertNotIn("final.json", uploaded_names)
                session.finish.assert_called_once_with(exit_code=0)
                with self.assertRaises(ValueError):
                    import_history(directory, Tracking(mode="online"))

    def test_identity_guards_and_interrupted_run_status(self):
        sdk, config = fake_sdk(), tiny_config()
        with tempfile.TemporaryDirectory() as temporary, patch.dict("sys.modules", wandb=sdk):
            tracker = Tracker(Tracking(mode="online", group="original"), config, temporary)
            with self.assertRaises(KeyboardInterrupt), tracker:
                raise KeyboardInterrupt()
            self.assertEqual(tracker.run.summary["training/status"], "interrupted")
            tracker.run.finish.assert_called_once_with(exit_code=1)
            for options in (Tracking(mode="online", group="different"),
                            Tracking(mode="online", entity="different"),
                            Tracking(mode="offline")):
                with self.assertRaises(ValueError), Tracker(options, config, temporary, resume=True):
                    pass
            self.assertEqual(sdk.init.call_count, 1)


if __name__ == "__main__":
    unittest.main()
