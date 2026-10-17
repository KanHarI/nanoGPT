"""Behavioral checks for arithmetic, distributions, causality, and exact resume."""

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.nn import functional as F

from model import GPT, GPTConfig
from experiments.finite_field.config import Config, gamma_schedule
from experiments.finite_field.data import (
    Example, Generator, IGNORE_INDEX, NUMERALS, PRIMES, STOI, VOCAB,
    batch, evaluate_rpn, operate,
)
from experiments.finite_field.evaluate import EvaluationSet, evaluate_checkpoint
from experiments.finite_field.train import run


class ArithmeticTests(unittest.TestCase):
    def test_every_single_operation(self):
        for p in PRIMES:
            for a in range(p):
                for b in range(p):
                    for op in "+-*/^":
                        expression = NUMERALS[a] + NUMERALS[b] + op
                        if op == "/" and b == 0:
                            with self.assertRaises(ZeroDivisionError):
                                operate(op, a, b, p)
                            with self.assertRaises(ZeroDivisionError):
                                evaluate_rpn(expression, p)
                        else:
                            self.assertEqual(operate(op, a, b, p), evaluate_rpn(expression, p))

    def test_wire_format_and_conventions(self):
        self.assertEqual(Example(7, "21+4/", 6, 2, 2).text, "7.21+4/=6\n")
        self.assertEqual(Example(19, "ii+", 17, 1, 1).text, "j.ii+=h\n")
        self.assertEqual(evaluate_rpn("00^", 19), 1)
        self.assertEqual(evaluate_rpn("23-", 7), 6)
        # Right subtree 2-3 is 6 in F7, so this is 3^6, not 3^(-1).
        self.assertEqual(evaluate_rpn("323-^", 7), 1)
        for invalid in ("", "12", "+", "j", "12?", "12++"):
            with self.assertRaises(ValueError):
                evaluate_rpn(invalid, 19)

    def test_generated_trees_and_operator_toggles(self):
        for include_power in (False, True):
            for include_division in (False, True):
                generator = Generator(10, max_operators=16, include_power=include_power,
                                      include_division=include_division)
                seen_power, seen_division = False, False
                for _ in range(500):
                    example = generator.sample(7)
                    self.assertEqual(example.answer, evaluate_rpn(example.expression, example.prime))
                    self.assertEqual(len(example.expression), 2 * example.operators + 1)
                    self.assertLess(example.answer, example.prime)
                    self.assertLessEqual(example.depth, example.operators)
                    self.assertEqual(example.text[1], ".")
                    seen_power |= "^" in example.expression
                    seen_division |= "/" in example.expression
                self.assertEqual(seen_power, include_power)
                self.assertEqual(seen_division, include_division)
                if not include_division:
                    self.assertEqual(generator.attempts, generator.items)

    def test_sampling_balance_size_and_replay(self):
        generator = Generator(731, max_operators=20, include_division=True)
        examples = [generator.sample(4) for _ in range(4000)]
        counts = Counter(e.prime for e in examples)
        self.assertEqual(set(counts), set(PRIMES))
        self.assertTrue(all(400 < count < 600 for count in counts.values()))
        for prime in PRIMES:
            sizes = [e.operators for e in examples if e.prime == prime]
            self.assertAlmostEqual(sum(sizes) / len(sizes), 4, delta=0.35)
        self.assertGreater(generator.attempts, generator.items)
        state = generator.state_dict()
        expected = [generator.sample(4) for _ in range(20)]
        restored = Generator(0, max_operators=20, include_division=True)
        restored.load_state_dict(state)
        self.assertEqual(expected, [restored.sample(4) for _ in range(20)])
        self.assertEqual(generator.state_dict(), restored.state_dict())
        self.assertTrue(all(generator.sample(0).operators == 0 for _ in range(30)))
        with self.assertRaises(ValueError):
            Generator(0, max_operators=20).load_state_dict(state)


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def tiny_config(self):
        return Config(steps=6, batch_size=8, block_size=16, max_operators=4,
                      n_layer=1, n_head=2, n_embd=16, dropout=0.1,
                      gamma_low=0, gamma_high=2, warmup_steps=1, eval_interval=6,
                      eval_per_field=2, final_per_field=2, baseline_per_field=4,
                      eval_gammas=(0, 2), log_interval=2, threads=1)

    def test_mask_and_causal_answer_position(self):
        examples = [Example(7, "21+4/", 6, 2, 2), Example(19, "i", 18, 0, 0)]
        x, y = batch(examples, 16)
        selected = y != IGNORE_INDEX
        self.assertEqual(selected.sum().item(), len(examples))
        for row, example in enumerate(examples):
            index = example.text.index("=")
            self.assertEqual(x[row, index].item(), STOI["="])
            self.assertEqual(y[row, index].item(), STOI[NUMERALS[example.answer]])
        torch.manual_seed(5)
        model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=16,
                              vocab_size=len(VOCAB), dropout=0.0))
        logits, loss = model(x, y)
        torch.testing.assert_close(loss, F.cross_entropy(logits[selected], y[selected]))
        changed = x.clone()
        for row, example in enumerate(examples):
            changed[row, example.text.index("=") + 1:] = STOI["j"]
        other_logits, _ = model(changed, y)
        torch.testing.assert_close(logits[selected], other_logits[selected], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            batch(examples, 4)

    def test_schedule_controls_and_validation(self):
        config = Config(steps=100, warmup_steps=2)
        curriculum = gamma_schedule(config)
        mixture = gamma_schedule(replace(config, condition="mixture"))
        self.assertEqual(curriculum[0], 1)
        self.assertEqual(curriculum[39:], [16.0] * 61)
        self.assertEqual(sorted(curriculum), sorted(mixture))
        self.assertNotEqual(curriculum, mixture)
        self.assertEqual(gamma_schedule(replace(config, condition="hard")), [16.0] * 100)
        self.assertEqual(mixture, gamma_schedule(replace(config, condition="mixture")))
        for invalid in (replace(config, block_size=8), replace(config, gamma_high=-1),
                        replace(config, include_power="false"), replace(config, include_division="false"),
                        replace(config, steps=0)):
            with self.assertRaises(ValueError):
                invalid.validate()

    def test_evaluation_balances_fields_and_preserves_rng(self):
        config = self.tiny_config()
        sampler = Generator(1)
        sampler_state, torch_state = sampler.state_dict(), torch.get_rng_state()
        evaluation = EvaluationSet(config, 2, 5, 100)
        self.assertTrue(all("/" not in e.expression for e in evaluation.examples))
        self.assertEqual(evaluation.generation_attempts, len(evaluation.examples))
        self.assertEqual(Counter(e.prime for e in evaluation.examples), {p: 5 for p in PRIMES})
        self.assertEqual(sampler.state_dict(), sampler_state)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=16, vocab_size=len(VOCAB)))
        model.train()
        torch_state = torch.get_rng_state()
        first = evaluation.evaluate(model, config)
        second = evaluation.evaluate(model, config)
        self.assertEqual(first, second)
        self.assertEqual(first["count"], 40)
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(torch.get_rng_state(), torch_state))
        self.assertAlmostEqual(first["loss"], sum(v["loss"] for v in first["by_field"].values()) / 8)

    def test_exact_cpu_resume_with_dropout(self):
        config = self.tiny_config()
        with tempfile.TemporaryDirectory() as temporary:
            full = Path(temporary) / "full"
            split = Path(temporary) / "split"
            run(config, full)
            run(config, split, stop_after=3)
            # Simulate a metrics write after the most recent saved checkpoint.
            with (split / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps({"kind": "uncommitted"}) + "\n")
            run(config, split, resume=True)
            a = torch.load(full / "checkpoint.pt", weights_only=False)
            b = torch.load(split / "checkpoint.pt", weights_only=False)
            self.assertEqual(a["step"], 6)
            self.assertEqual(a["sampler"], b["sampler"])
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)
            self.assertNotIn("uncommitted", (split / "metrics.jsonl").read_text())
            with self.assertRaises(FileExistsError):
                run(config, full)
            with self.assertRaises(ValueError):
                run(replace(config, steps=7), split, resume=True)

    def test_legacy_division_checkpoint_resume_and_evaluation(self):
        config = replace(self.tiny_config(), include_division=True)
        with tempfile.TemporaryDirectory() as temporary:
            full, split = Path(temporary) / "full", Path(temporary) / "legacy"
            run(config, full)
            run(config, split, stop_after=3)
            checkpoint = torch.load(split / "checkpoint.pt", weights_only=False)
            checkpoint["config"].pop("include_division")
            checkpoint["sampler"].pop("ops")
            torch.save(checkpoint, split / "checkpoint.pt")
            (split / "config.json").write_text(json.dumps(checkpoint["config"]))
            # Partial configs use today's default; historical run configs migrate.
            self.assertFalse(Config.load(split / "config.json").include_division)
            migrated = Config.load_saved(split / "config.json")
            self.assertTrue(migrated.include_division)
            report = evaluate_checkpoint(split / "checkpoint.pt")
            self.assertTrue(report["config"]["include_division"])
            expected = EvaluationSet(migrated, config.gamma_high, config.final_per_field,
                                     config.eval_seed + 10_000_019)
            self.assertEqual(report["evaluations"][0]["generation_attempts"], expected.generation_attempts)
            with self.assertRaises(ValueError):
                run(replace(migrated, include_division=False), split, resume=True)
            run(migrated, split, resume=True)
            a = torch.load(full / "checkpoint.pt", weights_only=False)
            b = torch.load(split / "checkpoint.pt", weights_only=False)
            self.assertEqual(a["sampler"], b["sampler"])
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            for name in a["model"]:
                torch.testing.assert_close(a["model"][name], b["model"][name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
