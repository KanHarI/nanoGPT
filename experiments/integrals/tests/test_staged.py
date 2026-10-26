from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.integrals.benchmark import staged_config
from experiments.integrals.config import Config, batch_plan
from experiments.integrals.corpus import Corpus, data_recipe, prepare
from experiments.integrals.data import example_plan
from .test_remote import fixture_generate, tiny


class StagedTests(unittest.TestCase):
    def test_equal_predicted_time_includes_final_stage(self):
        config = replace(Config(), gamma_high=3)
        chosen = staged_config(config, [(1, 1.), (2, 2.), (3, 4.)], 2)
        self.assertEqual(chosen.gamma_stage_steps, (40, 20, 10))
        self.assertEqual(chosen.steps, 70)
        self.assertEqual(chosen.ramp_steps, 70)
        self.assertEqual([s.gamma for s in batch_plan(chosen)], [1]*40+[2]*20+[3]*10)
        self.assertEqual([n*t for n,t in zip(chosen.gamma_stage_steps,(1,2,4))], [40]*3)
        self.assertEqual(chosen.generation_interval % chosen.eval_interval, 0)

    def test_stage_plan_preserves_exact_shuffled_samples_and_resume_identity(self):
        with tempfile.TemporaryDirectory() as directory, patch("experiments.integrals.corpus._generate", fixture_generate):
            config = replace(tiny(directory), steps=5, ramp_steps=5, gamma_stage_steps=(2,3))
            a = batch_plan(config)
            b = batch_plan(replace(config, condition="shuffled"))
            samples = lambda specs: {s.index:list(example_plan(s.gamma,s.seed,8,s.gamma_sampling)) for s in specs}
            self.assertEqual(samples(a), samples(b))
            root = Path(directory)
            prepare(config, root/"data", conditions=["curriculum","shuffled"])
            Corpus(root/"data", replace(config,condition="shuffled"), verify_all=True)
            with self.assertRaises(ValueError):
                Corpus(root/"data", replace(config,gamma_stage_steps=(3,2)))
            self.assertEqual(data_recipe(config)["gamma_stage_steps"],[2,3])
            self.assertNotIn("gamma_stage_steps",Config().to_dict())

    def test_invalid_stage_budgets(self):
        for kwargs in (dict(gamma_stage_steps=(2,2)),
                       dict(steps=4,ramp_steps=4,gamma_high=2,gamma_stage_steps=(0,4)),
                       dict(steps=4,ramp_steps=4,gamma_high=2.5,gamma_stage_steps=(2,2))):
            with self.assertRaises(ValueError):
                replace(Config(),warmup_steps=1,**kwargs).validate()
        for timings,minutes in (([(1,1.)],120), ([(g,0.) for g in range(1,21)],120),
                                ([(g,1.) for g in range(1,21)],0)):
            with self.assertRaises(ValueError):
                staged_config(Config(),timings,minutes)


if __name__ == "__main__":
    unittest.main()
