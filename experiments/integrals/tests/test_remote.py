from dataclasses import replace
import json
import shutil
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from experiments.integrals.benchmark import calibrated_config, choose_candidate
from experiments.integrals.config import Config, batch_plan, ramp_updates
from experiments.integrals.corpus import Corpus, Prefetch, _generate, prepare, tasks
from experiments.integrals.data import Dataset, stream_seed
from experiments.integrals.evaluate import Evaluation, baseline_scores, fit_bit_prior
from experiments.integrals.sweep import Archive, completed_report, sweep
from experiments.integrals.train import run
from .test_integrals import ScriptedModel, example


def tiny(directory):
    return replace(Config(), steps=4, warmup_steps=1, ramp_steps=2, batch_size=2,
                   n_layer=1, n_embd=16, dropout=.1, max_operators=2,
                   gamma_high=2, eval_gammas=(1,2), eval_examples=2, final_examples=2,
                   baseline_examples=2, generation_examples=1, final_generation_examples=2,
                   eval_interval=1, generation_interval=2, checkpoint_interval=2,
                   log_interval=1, threads=1, cache_dir=str(Path(directory)/"cache"))


def fixture_generate(payload):
    task, _ = payload
    return task, [(example((task["seed"]+i)%3).to_dict(), {"method":"test-fixture"})
                  for i in range(task["count"])], {"generated":task["count"]}


class CorpusTests(unittest.TestCase):
    def test_export_resume_checksums_and_identical_shuffled_examples(self):
        with tempfile.TemporaryDirectory() as directory, patch("experiments.integrals.corpus._generate", fixture_generate):
            config = tiny(directory)
            path = Path(directory)/"data"
            manifest = prepare(config, path, tasks_per_shard=3)
            original = manifest.read_bytes()
            with patch("experiments.integrals.corpus._generate", side_effect=AssertionError("unexpected regeneration")):
                self.assertEqual(prepare(config, path, tasks_per_shard=3).read_bytes(), original)
            corpus = Corpus(path, config, verify_all=True)
            curriculum = {spec.index:[e.text for e in corpus.examples(spec.gamma,spec.seed,config.batch_size)]
                          for spec in batch_plan(config)}
            for spec in batch_plan(replace(config,condition="shuffled")):
                self.assertEqual([e.text for e in corpus.examples(spec.gamma,spec.seed,config.batch_size)], curriculum[spec.index])
            prefetched = list(Prefetch(corpus, batch_plan(config), config.batch_size))
            self.assertEqual([spec.index for spec,_ in prefetched], list(range(4)))
            with self.assertRaises(ValueError):
                corpus.examples(99, 999, 1)
            with self.assertRaises(ValueError):
                Corpus(path, replace(config, batch_size=3))
            shard = next(path.glob("part-*/tokens.bin"))
            data = bytearray(shard.read_bytes())
            data[0] ^= 1
            shard.write_bytes(data)
            with self.assertRaises(ValueError):
                Corpus(path, config, verify_all=True)

    def test_interrupted_export_reuses_only_committed_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny(directory)
            path = Path(directory)/"data"
            calls = []
            def fail(payload):
                calls.append(payload[0]["name"])
                if len(calls) == 3:
                    raise RuntimeError("simulated worker failure")
                return fixture_generate(payload)
            with patch("experiments.integrals.corpus._generate", fail), self.assertRaises(RuntimeError):
                prepare(config, path, tasks_per_shard=2)
            self.assertFalse((path/"manifest.json").exists())
            first = (path/"part-000000/shard.json").read_bytes()
            retried = []
            def record(payload):
                retried.append(payload[0]["name"])
                return fixture_generate(payload)
            with patch("experiments.integrals.corpus._generate", record):
                prepare(config, path, tasks_per_shard=2)
            self.assertEqual(first, (path/"part-000000/shard.json").read_bytes())
            self.assertNotIn(calls[0], retried)
            self.assertNotIn(calls[1], retried)
            Corpus(path, config, verify_all=True)

    def test_real_labels_match_procedural_identity_and_stress_length(self):
        with tempfile.TemporaryDirectory() as directory:
            config = tiny(directory)
            task = tasks(config)[0]
            settings = dict(max_operators=config.max_operators, timeout_seconds=5, max_attempts=100)
            _, rows, _ = _generate((task, settings))
            direct = Dataset(None, **settings).examples(task["gamma"], task["seed"], task["count"])
            self.assertEqual([row[0] for row in rows], [e.to_dict() for e in direct])
            stress = dict(name="pilot/stress", gamma=20, seed=0, count=1)
            _, rows, _ = _generate((stress, {**settings, "max_operators":32}))
            from experiments.integrals.data import Example
            self.assertEqual(len(Example.from_dict(rows[0][0]).tokens), 86)


class CalibrationTests(unittest.TestCase):
    def test_selection_requires_headroom_and_prefers_small_near_best_batch(self):
        rows = [dict(status="ok",batch_size=64,memory_fraction=.4,examples_per_second=96),
                dict(status="ok",batch_size=128,memory_fraction=.7,examples_per_second=100),
                dict(status="ok",batch_size=256,memory_fraction=.95,examples_per_second=200)]
        self.assertEqual(choose_candidate(rows)["batch_size"],64)
        with self.assertRaises(RuntimeError):
            choose_candidate(rows[2:])

    def test_schedule_targets_each_phase_and_decouples_evaluation(self):
        config = Config()
        chosen = calibrated_config(config,[(1,1.),(20,3.)],1,1,2.,4.)
        self.assertEqual(ramp_updates(chosen),30)
        self.assertEqual(chosen.steps,50)
        self.assertEqual(batch_plan(chosen)[29].gamma,20)
        self.assertEqual(chosen.generation_interval % chosen.eval_interval,0)
        self.assertGreater(chosen.generation_interval,chosen.eval_interval)
        self.assertGreater(chosen.checkpoint_interval,0)
        with self.assertRaises(ValueError):
            calibrated_config(config,[(1,0.),(20,3.)],1,1,2.,4.)


class EvaluationAndRunnerTests(unittest.TestCase):
    def test_generation_batches_across_gamma_and_split_banks(self):
        class Fixture:
            def examples(self,gamma,seed,count):
                return [example()]*count
        config = replace(Config(),eval_gammas=(1,20),eval_examples=1,final_examples=1,
                         generation_examples=1,baseline_examples=1,eval_batch_size=8)
        evaluation = Evaluation(config, Fixture())
        model = ScriptedModel(example().answer,len(example().prefix)+1)
        with patch("experiments.integrals.evaluate.evaluate_examples", return_value={"answer_nll":1.}):
            result = evaluation.evaluate(model,config)
            self.assertEqual(len(model.prompts),16)  # One 4-example group, not four decoding loops.
            self.assertTrue(all(len(prompt)==4 for prompt in model.prompts))
            self.assertTrue(all(cell["exact_answer_accuracy"]==1 for split in result.values() for cell in split.values()))
            model.prompts.clear()
            evaluation.evaluate(model,config,include_generation=False)
            self.assertEqual(model.prompts,[])

    def test_baseline_is_smoothed_and_zero_does_not_get_credit_for_nonzero_answers(self):
        probabilities = fit_bit_prior([example()]*4)
        self.assertTrue(all(0<p<1 for p in probabilities))
        result = baseline_scores([example()],probabilities,Config())
        self.assertLess(result["answer_bit_prior_nll"], .2)
        self.assertEqual(result["always_zero_exact_accuracy"],0)
        self.assertEqual(result["always_zero_tolerance_accuracy"],0)

    def test_prepared_training_resume_and_durable_snapshot(self):
        with tempfile.TemporaryDirectory() as directory, patch("experiments.integrals.corpus._generate",fixture_generate):
            root = Path(directory)
            config = tiny(root)
            prepare(config,root/"data",tasks_per_shard=3)
            fingerprint = Corpus(root/"data",config).fingerprint
            run(config,root/"full",data_dir=root/"data")
            archive = Archive(root/"archive",root/"partial")
            run(config,root/"partial",stop_after=2,data_dir=root/"data",checkpoint_callback=archive.snapshot)
            archived = archive.verify("partial",completed=False)
            shutil.copytree(archived,root/"recovered")
            run(config,root/"partial",resume=True,data_dir=root/"data",checkpoint_callback=archive.snapshot)
            recovered_archive = Archive(root/"recovered-archive",root/"recovered")
            run(config,root/"recovered",resume=True,data_dir=root/"data",checkpoint_callback=recovered_archive.snapshot)
            recovered_archive.verify("recovered")
            archive.verify("partial")
            a = torch.load(root/"full/checkpoint.pt",weights_only=False)
            b = torch.load(root/"partial/checkpoint.pt",weights_only=False)
            recovered = torch.load(root/"recovered/checkpoint.pt",weights_only=False)
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]),key)
                self.assertTrue(torch.equal(a["model"][key], recovered["model"][key]),key)
            completed_report(root/"partial",config,fingerprint)
            self.assertLessEqual(len(list((root/"archive/partial").glob("snapshot-*"))),2)
            records = [json.loads(s) for s in (root/"full/metrics.jsonl").read_text().splitlines()]
            at_one = next(r for r in records if r["kind"]=="eval" and r["step"]==1)
            at_two = next(r for r in records if r["kind"]=="eval" and r["step"]==2)
            self.assertNotIn("valid_answer_rate",at_one["loss_grid"]["test"]["2"])
            self.assertIn("valid_answer_rate",at_two["loss_grid"]["test"]["2"])
            final = json.loads((root/"full/final.json").read_text())
            self.assertEqual(final["loss_grid"]["test"]["2"]["generation_examples"],2)
            # A corrupted durable artifact cannot pass the shutdown gate.
            latest = archive.verify("partial")
            (latest/"config.json").write_text("broken")
            with self.assertRaises(ValueError):
                archive.verify("partial")

    def test_sweep_resumes_completed_conditions_and_never_shuts_down_by_default(self):
        with tempfile.TemporaryDirectory() as directory, patch("experiments.integrals.corpus._generate",fixture_generate), \
                patch("experiments.integrals.sweep.shutdown_host") as shutdown:
            root = Path(directory)
            config = tiny(root)
            prepare(config,root/"data",tasks_per_shard=3)
            result = sweep(config,root/"data",root/"run",archive_dir=root/"archive")
            self.assertEqual(set(result["conditions"]),{"curriculum","shuffled","hard"})
            with patch("experiments.integrals.sweep.run",side_effect=AssertionError("unexpected retraining")):
                self.assertEqual(sweep(config,root/"data",root/"run",resume=True,archive_dir=root/"archive"),result)
            shutdown.assert_not_called()
            with self.assertRaises(ValueError):
                sweep(config,root/"data",root/"elsewhere",shutdown_on_success=True)

    def test_tracking_finish_failure_cannot_mark_run_complete(self):
        with tempfile.TemporaryDirectory() as directory, patch("experiments.integrals.corpus._generate",fixture_generate):
            root = Path(directory)
            config = tiny(root)
            prepare(config,root/"data",tasks_per_shard=3)
            with patch("experiments.integrals.tracking.Tracker.__exit__",side_effect=RuntimeError("SDK flush failed")):
                with self.assertRaises(RuntimeError):
                    run(config,root/"run",data_dir=root/"data")
            checkpoint = torch.load(root/"run/checkpoint.pt",weights_only=False)
            self.assertFalse(checkpoint["completed"])
            self.assertFalse((root/"run/tracking_complete.json").exists())
            with self.assertRaises(ValueError):
                completed_report(root/"run",config,Corpus(root/"data",config).fingerprint)


if __name__ == "__main__":
    unittest.main()
