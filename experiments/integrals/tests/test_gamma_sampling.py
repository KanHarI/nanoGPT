from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.integrals.config import BatchSpec, Config, batch_plan
from experiments.integrals.corpus import Corpus, _generate, data_recipe, prepare, tasks
from experiments.integrals.data import Dataset, example_plan
from .test_remote import fixture_generate, tiny


class GammaSamplingTests(unittest.TestCase):
    def test_independent_uniform_integer_draws_and_unchanged_ast_seeds(self):
        plan = list(example_plan(20, 123, 20000, "uniform_integer"))
        counts = Counter(g for g,_ in plan)
        self.assertEqual(set(counts), set(range(1, 21)))
        self.assertTrue(all(800 < n < 1200 for n in counts.values()))
        self.assertGreater(len({g for g,_ in plan[:16]}), 1)  # Per sample, not per batch.
        self.assertEqual(plan[:16], list(example_plan(20, 123, 16, "uniform_integer")))
        self.assertEqual([s for _,s in plan], [s for _,s in example_plan(20, 123, 20000)])
        for ceiling in (1, 1.9, 5.7):
            draws = [g for g,_ in example_plan(ceiling, 4, 1000, "uniform_integer")]
            self.assertEqual(set(draws), set(range(1, int(ceiling)+1)))
        self.assertEqual({g for g,_ in example_plan(5.7, 4, 10)}, {5.7})

    def test_invalid_ceiling_and_mode(self):
        for gamma,mode in ((0,"uniform_integer"), (.9,"uniform_integer"),
                           (float("nan"),"uniform_integer"), (5,"unknown")):
            with self.assertRaises(ValueError):
                list(example_plan(gamma, 0, 1, mode))
        with self.assertRaises(ValueError):
            replace(Config(), gamma_low=0).validate()

    def test_shuffled_preserves_all_sample_draws_and_hard_stays_fixed(self):
        config = replace(Config(), steps=30, warmup_steps=2)
        plans = {condition:batch_plan(replace(config, condition=condition))
                 for condition in ("curriculum", "shuffled", "hard")}
        def samples(specs):
            return {s.index:list(example_plan(s.gamma,s.seed,64,s.gamma_sampling)) for s in specs}
        self.assertEqual(samples(plans["curriculum"]), samples(plans["shuffled"]))
        self.assertEqual({g for rows in samples(plans["hard"]).values() for g,_ in rows}, {20})

    def test_legacy_configs_preserve_serialized_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"config.json"
            old = replace(Config(), training_gamma_sampling="fixed").to_dict()
            self.assertNotIn("training_gamma_sampling", old)
            path.write_text(json.dumps(old))
            loaded = Config.load(path)
            self.assertEqual(loaded.training_gamma_sampling, "fixed")
            self.assertEqual(loaded.to_dict(), old)
            self.assertNotIn("training_gamma_sampling", data_recipe(loaded))
            current = Config().to_dict()
            path.write_text(json.dumps(current))
            self.assertEqual(Config.load(path).to_dict(), current)

    def test_preparation_policy_is_explicit_and_eval_stays_fixed(self):
        with tempfile.TemporaryDirectory() as directory, patch("experiments.integrals.corpus._generate", fixture_generate):
            config = tiny(directory)
            for task in tasks(config):
                expected = "uniform_integer" if task["name"].startswith("training/curriculum/") else "fixed"
                self.assertEqual(task.get("gamma_sampling", "fixed"), expected)
            for task in tasks(config, kind="pilot"):
                expected = "uniform_integer" if task["name"].startswith("pilot/") and task["name"] != "pilot/stress" else "fixed"
                self.assertEqual(task.get("gamma_sampling", "fixed"), expected)
            path = Path(directory)/"data"
            prepare(config, path)
            corpus = Corpus(path, config)
            spec = batch_plan(config)[-1]
            rows = corpus.training_examples(spec, config.batch_size)
            self.assertEqual([r.gamma for r in rows],
                             [g for g,_ in example_plan(spec.gamma,spec.seed,config.batch_size,spec.gamma_sampling)])
            with self.assertRaises(ValueError):
                corpus.examples(spec.gamma,spec.seed,config.batch_size)  # Cannot read mixed training as fixed gamma.
            with self.assertRaises(ValueError):
                Corpus(path, replace(config, training_gamma_sampling="fixed"))
            with self.assertRaises(ValueError):
                prepare(replace(config, training_gamma_sampling="fixed"), path)
            # The old fixed-gamma export format also remains readable.
            legacy = replace(config, training_gamma_sampling="fixed")
            prepare(legacy, Path(directory)/"legacy")
            Corpus(Path(directory)/"legacy", legacy, verify_all=True)

    def test_prepared_and_inline_real_labels_match_per_sample(self):
        settings = dict(max_operators=2, timeout_seconds=5, max_attempts=100)
        task = dict(name="training/curriculum/0", gamma=5.7, seed=4321,
                    count=12, gamma_sampling="uniform_integer")
        _, rows, _ = _generate((task, settings))
        spec = BatchSpec(0, task["gamma"], task["seed"], task["gamma_sampling"])
        direct = Dataset(None, **settings).training_examples(spec, task["count"])
        self.assertEqual([r[0] for r in rows], [e.to_dict() for e in direct])
        self.assertGreater(len({e.gamma for e in direct}), 1)
        for (row, provenance), (gamma, seed) in zip(rows, example_plan(5.7,4321,12,"uniform_integer")):
            self.assertEqual(row["gamma"], gamma)
            self.assertEqual(provenance["identity"]["gamma"], gamma)
            self.assertEqual(provenance["identity"]["seed"], seed)


if __name__ == "__main__":
    unittest.main()
