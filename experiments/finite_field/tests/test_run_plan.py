"""Explicitly stopped conditions must stay excluded from recovery/comparison."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from experiments.finite_field.config import Config
from experiments.finite_field.run_plan import active_conditions
from experiments.finite_field.sweep import completed_run
from experiments.finite_field.watchdog import inspect_runs


class RunPlanTests(unittest.TestCase):
    def test_monitor_and_sweep_share_the_active_conditions(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'run_plan.json').write_text(json.dumps(dict(conditions=['curriculum', 'mixture'])))
            self.assertEqual(active_conditions(root), ('curriculum', 'mixture'))
            for condition in ('curriculum', 'mixture'):
                path = root / f'{condition}-seed0'
                path.mkdir()
                config = Config(condition=condition).to_dict()
                (path / 'config.json').write_text(json.dumps(config))
                (path / 'final.json').write_text(json.dumps(dict(step=config['steps'], config=config)))
            (root / 'hard-seed0').mkdir()
            (root / 'comparison').mkdir()
            (root / 'comparison' / 'summary.json').write_text(json.dumps(
                dict(conditions={'curriculum': {}, 'mixture': {}}, seeds=[0])))
            status = inspect_runs(root)
            self.assertTrue(status['all_complete'])
            self.assertTrue(status['comparison_complete'])
            self.assertEqual([r['condition'] for r in status['runs']], ['curriculum', 'mixture'])

    def test_skip_complete_requires_matching_full_budget_report_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = Config()
            report = dict(step=config.steps, config=config.to_dict())
            (root / 'final.json').write_text(json.dumps(report))
            self.assertFalse(completed_run(root, config))
            (root / 'checkpoint.pt').touch()
            self.assertTrue(completed_run(root, config))
            self.assertFalse(completed_run(root, replace(config, seed=1)))
            report['step'] -= 1
            (root / 'final.json').write_text(json.dumps(report))
            self.assertFalse(completed_run(root, config))


if __name__ == '__main__':
    unittest.main()
