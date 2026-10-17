"""Packing boundaries, answer causality, paired loss banks, and batch selection."""

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from experiments.finite_field.benchmark import choose_candidate
from experiments.finite_field.config import learning_rate, Config
from experiments.finite_field.data import Example, PRIMES, VOCAB, STOI, IGNORE_INDEX, batch
from experiments.finite_field.evaluate import LossGrid
from experiments.finite_field.tests.test_tracking import tiny_config
from experiments.finite_field.tracking import metrics, grid_metrics
from experiments.finite_field.train import run


class GpuPipelineTests(unittest.TestCase):
    def test_constant_lr_after_warmup_and_legacy_cosine(self):
        config = replace(tiny_config(), warmup_steps=2, lr_schedule="constant")
        self.assertEqual(learning_rate(config, 0), config.learning_rate / 2)
        for step in range(1, config.steps):
            self.assertEqual(learning_rate(config, step), config.learning_rate)
        legacy = config.to_dict()
        legacy.pop("lr_schedule")
        restored = Config.from_saved(legacy)
        self.assertEqual(restored.lr_schedule, "cosine")
        self.assertAlmostEqual(learning_rate(restored, restored.steps - 1),
                               restored.learning_rate * restored.min_lr_ratio)

    def test_packing_keeps_whole_lines_and_answer_causality(self):
        examples = [Example(7, "21+4/", 6, 2, 2), Example(19, "i", 18, 0, 0),
                    Example(2, "1", 1, 0, 0)]
        x, y = batch(examples, 16, pack=True)
        self.assertEqual(x.shape, (2, 16))
        self.assertEqual("".join(VOCAB[t] for t in x[0].tolist()), "7.21+4/=6\nj.i=i\n")
        self.assertEqual((y != IGNORE_INDEX).sum().item(), 3)
        self.assertEqual(y[y != IGNORE_INDEX].tolist(), [STOI['6'], STOI['i'], STOI['1']])
        self.assertTrue(all(row[0].item() in [STOI[str(p)] for p in (2, 7)] for row in x))
        model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=16,
                              vocab_size=len(VOCAB), dropout=0, bias=False))
        logits, loss = model(x, y)
        selected = y != IGNORE_INDEX
        torch.testing.assert_close(loss, F.cross_entropy(logits[selected], y[selected]))
        # Change each answer and all later tokens independently. Its own target
        # prediction must be unaffected, even in a context with multiple items.
        for row, column in selected.nonzero().tolist():
            changed = x.clone()
            changed[row, column + 1:] = STOI['j']
            other, _ = model(changed, y)
            torch.testing.assert_close(logits[row, column], other[row, column], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            batch(examples, 5, pack=True)

    def test_loss_grid_is_balanced_independent_repeatable_and_loss_only(self):
        config = replace(tiny_config(), eval_mode="loss_grid", pack_expressions=True,
                         eval_gammas=tuple(range(1, 13)), gamma_high=12)
        state = torch.get_rng_state()
        grid = LossGrid(config)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertEqual(len(grid.banks), 24)
        for batches in grid.banks.values():
            fields = [p for _, _, chunk in batches for p in chunk]
            self.assertEqual({p: fields.count(p) for p in PRIMES}, {p: 2 for p in PRIMES})
        self.assertFalse(torch.equal(grid.banks['train', 1][0][0], grid.banks['test', 1][0][0]))
        model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=16,
                              vocab_size=len(VOCAB), dropout=0.1, bias=False))
        model.train()
        state = torch.get_rng_state()
        first = grid.evaluate(model, config)
        second = grid.evaluate(model, config)
        self.assertEqual(first, second)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        logged = grid_metrics(first, 4)
        self.assertEqual(len(logged), 193)  # 2 streams x 12 gammas x 8 fields + x-axis
        self.assertEqual(logged['eval/gamma_01/field_19/test_loss'], first['test']['1']['by_field']['19']['loss'])
        self.assertTrue(all(key == 'optimizer_step' or key.endswith(('/train_loss', '/test_loss')) for key in logged))
        self.assertIsNone(metrics(dict(kind='train', step=4, gamma=2, loss=3), loss_only=True))

    def test_packed_training_resume_and_final_grid(self):
        config = replace(tiny_config(), pack_expressions=True, eval_mode="loss_grid")
        with tempfile.TemporaryDirectory() as temporary:
            full, split = Path(temporary) / 'full', Path(temporary) / 'split'
            run(config, full, final_evaluation=True)
            run(config, split, stop_after=3)
            run(config, split, resume=True, final_evaluation=True)
            a = torch.load(full / 'checkpoint.pt', weights_only=False)
            b = torch.load(split / 'checkpoint.pt', weights_only=False)
            self.assertEqual(a['sampler'], b['sampler'])
            for name in a['model']:
                torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0)
            report = json.loads((full/'final.json').read_text())
            self.assertEqual(set(report['loss_grid']), {'train', 'test'})
            self.assertEqual(report['evaluations'][0]['seed'], config.eval_seed + 10_000_019)

    def test_batch_selection_respects_headroom_and_avoids_unhelpful_large_batch(self):
        results = [dict(status='ok', batch_size=b, answers_per_second=s, memory_fraction=m)
                   for b, s, m in [(64, 80, .2), (128, 97, .4), (256, 100, .7), (512, 150, .95)]]
        self.assertEqual(choose_candidate(results)['batch_size'], 128)
        with self.assertRaises(RuntimeError):
            choose_candidate([dict(status='oom', batch_size=32)])


if __name__ == '__main__':
    unittest.main()
