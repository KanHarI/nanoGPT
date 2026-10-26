"""Arithmetic, sampling, exact exposure matching, causal masking, and resume."""

from collections import Counter
from contextlib import contextmanager
from dataclasses import replace
import itertools
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import unittest

import torch
from torch.nn import functional as F

from experiments.finite_field.tests.test_tracking import fake_sdk
from experiments.minmax.config import Config, batch_plan, generate_batch, learning_rate
from experiments.minmax.data import (ARITY, DECODE, ENCODE, NUMERALS, OPS, STOI, VOCAB,
                                     Generator, IGNORE_INDEX, Node, batch, evaluate_rpn,
                                     example_from_ast, parse_rpn, split_counts, stream_seed)
from experiments.minmax.evaluate import Evaluation
from experiments.minmax.sweep import compare, completed
from experiments.minmax.tracking import Tracking, grid_metrics
from experiments.minmax.train import make_model, run, write_json


def tiny_config():
    return Config(steps=6, batch_size=8, block_size=16, max_operators=4,
                  n_layer=1, n_head=2, n_embd=16, dropout=0.1, warmup_steps=2,
                  gamma_high=2, eval_interval=6, eval_examples=8, final_examples=12,
                  baseline_examples=16, eval_batch_size=8, eval_gammas=(1, 2),
                  log_interval=1, threads=1)


@contextmanager
def mock_wandb(sdk):
    # Restore only this entry. patch.dict(sys.modules) also removes lazy Torch
    # imports, whose C++ operator registrations cannot safely be initialized twice.
    missing = object()
    previous = sys.modules.get("wandb", missing)
    sys.modules["wandb"] = sdk
    try:
        yield
    finally:
        if previous is missing:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = previous


class DataTests(unittest.TestCase):
    def test_encoding_and_exhaustive_arithmetic(self):
        self.assertEqual(len(VOCAB), len(set(VOCAB)))
        self.assertEqual(len(VOCAB), 48)
        self.assertEqual(set(ENCODE), set(range(-20, 21)))
        for value in range(-20, 21):
            self.assertEqual(DECODE[ENCODE[value]], value)
            self.assertEqual(evaluate_rpn(ENCODE[value] + "~"), -value)
        for left, right in itertools.product(range(-20, 21), repeat=2):
            prefix = ENCODE[left] + ENCODE[right]
            self.assertEqual(evaluate_rpn(prefix + "m"), min(left, right))
            self.assertEqual(evaluate_rpn(prefix + "n"), max(left, right))
        for values in itertools.product(range(-20, 21), repeat=3):
            prefix = "".join(ENCODE[v] for v in values)
            self.assertEqual(evaluate_rpn(prefix + "t"), min(values))
            self.assertEqual(evaluate_rpn(prefix + "u"), max(values))

    def test_examples_and_rejection_of_malformed_rpn(self):
        cases = {"120u312um": 2, "C2m": -3, "3~": -3, "T~": 20,
                 "3~2n": 2, "A02t": -1, "01n23nm": 1, "03n12nm": 2}
        for expression, answer in cases.items():
            node = parse_rpn(expression)
            self.assertEqual(node.render(), expression)
            self.assertEqual(node.evaluate(), answer)
            self.assertEqual(evaluate_rpn(expression), answer)
        for expression in ("", "12", "m", "1m", "12t", "0~~1", "1?", "1=1"):
            with self.assertRaises(ValueError):
                evaluate_rpn(expression)
            with self.assertRaises(ValueError):
                parse_rpn(expression)
        with self.assertRaises(ValueError):
            Node("t", (Node("1"), Node("2")))

    def test_stars_and_bars_is_bijective(self):
        class FixedBars:
            def __init__(self, bars):
                self.bars = bars
            def sample(self, population, count):
                return self.bars
        results = [split_counts(3, 3, FixedBars(bars)) for bars in itertools.combinations(range(5), 2)]
        expected = [(a, b, 3-a-b) for a in range(4) for b in range(4-a)]
        self.assertEqual(set(results), set(expected))
        self.assertEqual(len(results), len(set(results)))
        self.assertEqual(split_counts(7, 1, random.Random(0)), (7,))

    def test_generation_size_distribution_and_no_alternation_filter(self):
        generator = Generator(2026, max_operators=3)
        examples = [generator.sample(8) for _ in range(4000)]
        weights = [8**k / math.factorial(k) for k in range(4)]
        expected_mean = sum(k*w for k, w in enumerate(weights)) / sum(weights)
        self.assertAlmostEqual(sum(e.operators for e in examples) / len(examples), expected_mean, delta=0.05)
        roots = Counter(e.root for e in examples if e.operators)
        for op in OPS:
            self.assertAlmostEqual(roots[op] / sum(roots.values()), 0.2, delta=0.04)
        leaves = Counter(value for e in examples for value in e.leaves)
        for value in range(-20, 21):
            self.assertAlmostEqual(leaves[value] / sum(leaves.values()), 1/41, delta=0.01)
        for example in examples[:300]:
            self.assertEqual(example.answer, evaluate_rpn(example.expression))
        # Force one operator family: same-operator edges and double negations stay valid.
        for index, op in enumerate(OPS):
            weights = tuple(int(i == index) for i in range(5))
            tree = Generator(5, operator_weights=weights).tree(12)
            example = example_from_ast(tree)
            self.assertEqual(example.operators, 12)
            self.assertEqual(set(c for c in example.expression if c in ARITY), {op})
            self.assertEqual(example.answer, evaluate_rpn(example.expression))
            self.assertLessEqual(len(example.text), 3 * 12 + 4)
        ternary = example_from_ast(Generator(3, operator_weights=(0, 0, 0, 1, 0)).tree(32))
        self.assertEqual(len(ternary.text), 100)
        self.assertEqual((batch([ternary], 99)[1] != IGNORE_INDEX).sum(), 1)

    def test_depth_stack_and_negation_alternations(self):
        left = example_from_ast(parse_rpn("01m2m3m"))
        right = example_from_ast(parse_rpn("0123mmm"))
        self.assertEqual(left.depth, right.depth)
        self.assertEqual((left.stack_depth, right.stack_depth), (2, 4))
        self.assertEqual(left.alternations, 0)
        mixed = example_from_ast(parse_rpn("01m~2n"))
        self.assertEqual((mixed.alternations, mixed.negations), (1, 1))


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_packing_answer_mask_and_causality(self):
        examples = [example_from_ast(parse_rpn(text)) for text in ("C2m", "3~", "T~", "A02t")]
        x, y = batch(examples, 16)
        self.assertEqual("".join(VOCAB[i] for i in x[0].tolist()), "C2m=C\n3~=C\nT~=k\n")
        self.assertEqual(y[y != IGNORE_INDEX].tolist(), [STOI[ENCODE[e.answer]] for e in examples])
        config = replace(tiny_config(), dropout=0)
        model = make_model(config)
        logits, loss = model(x, y)
        selected = y != IGNORE_INDEX
        torch.testing.assert_close(loss, F.cross_entropy(logits[selected], y[selected]))
        for row, column in selected.nonzero().tolist():
            self.assertEqual(x[row, column].item(), STOI["="])
            changed = x.clone()
            changed[row, column+1:] = STOI["T"]
            altered, _ = model(changed, y)
            torch.testing.assert_close(logits[row, column], altered[row, column], rtol=0, atol=0)
        with self.assertRaises(ValueError):
            batch(examples, 3)

    def test_exact_example_replay_and_ramp_only_control(self):
        config = replace(tiny_config(), steps=20, ramp_fraction=0.4)
        ordered = batch_plan(config)
        shuffled = batch_plan(replace(config, condition="shuffled"))
        ramp = batch_plan(replace(config, condition="shuffled_ramp"))
        self.assertNotEqual(ordered, shuffled)
        self.assertEqual(sorted(shuffled, key=lambda s: s.index), ordered)
        self.assertEqual(ramp[8:], ordered[8:])
        self.assertEqual(sorted(ramp[:8], key=lambda s: s.index), ordered[:8])
        self.assertEqual((ordered[0].gamma, ordered[7].gamma), (1, 2))
        for spec in shuffled:
            self.assertEqual(generate_batch(config, ordered[spec.index]), generate_batch(config, spec))
        self.assertTrue(all(s.gamma == 2 for s in batch_plan(replace(config, condition="hard"))))
        self.assertEqual(learning_rate(config, 0), config.learning_rate / 2)
        for step in range(1, config.steps):
            self.assertEqual(learning_rate(config, step), config.learning_rate)

    def test_configuration_guards(self):
        config = tiny_config()
        for overrides in ({"block_size": 14}, {"operator_weights": (0,)*5}, {"n_head": 3},
                          {"learning_rate": float("nan")}, {"gamma_high": -1},
                          {"warmup_steps": config.steps}, {"version": "other"}):
            with self.assertRaises(ValueError):
                replace(config, **overrides).validate()

    def test_independent_repeatable_banks_and_loss_only_tracking(self):
        config = tiny_config()
        state = torch.get_rng_state()
        evaluation = Evaluation(config)
        final = Evaluation(config, final=True)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertTrue(set(evaluation.seeds.values()).isdisjoint(final.seeds.values()))
        self.assertEqual(len(set(evaluation.seeds.values())), 4)
        model = make_model(config)
        model.train()
        state = torch.get_rng_state()
        first = evaluation.evaluate(model, config)
        self.assertEqual(first, evaluation.evaluate(model, config))
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        for split in first.values():
            for result in split.values():
                self.assertEqual(result["count"], config.eval_examples)
                self.assertEqual(sum(v["count"] for v in result["by_root"].values()), config.eval_examples)
                self.assertAlmostEqual(result["baselines"]["uniform_loss"], math.log(41))
        values = grid_metrics(first, 3)
        self.assertEqual(set(values), {"optimizer_step", "eval/gamma_01/train_loss", "eval/gamma_01/test_loss",
                                       "eval/gamma_02/train_loss", "eval/gamma_02/test_loss"})

    def test_exact_resume_with_dropout_and_tracking(self):
        config, sdk = tiny_config(), fake_sdk()
        with tempfile.TemporaryDirectory() as temporary, mock_wandb(sdk):
            full, split = Path(temporary) / "full", Path(temporary) / "split"
            run(config, full)
            options = Tracking(mode="online", group="unit-test")
            run(config, split, stop_after=3, tracking=options)
            identity = json.loads((split / "wandb_run.json").read_text())
            with (split / "metrics.jsonl").open("a") as handle:
                handle.write('{"kind": "uncommitted"}\n')
            run(config, split, resume=True, tracking=options)
            self.assertEqual(json.loads((split / "wandb_run.json").read_text()), identity)
            self.assertEqual(sdk.init.call_args.kwargs["resume"], "must")
            a = torch.load(full / "checkpoint.pt", weights_only=False)
            b = torch.load(split / "checkpoint.pt", weights_only=False)
            for name in a["model"]:
                self.assertTrue(torch.equal(a["model"][name], b["model"][name]), name)
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            self.assertEqual(a["characters"], b["characters"])
            final = json.loads((split / "final.json").read_text())
            self.assertEqual(final["loss_grid"], json.loads((full / "final.json").read_text())["loss_grid"])
            self.assertNotIn("uncommitted", (split / "metrics.jsonl").read_text())
            self.assertTrue(completed(split, config))
            sdk.init.reset_mock()
            with self.assertRaises(ValueError):
                run(replace(config, steps=8), split, resume=True, tracking=options)
            sdk.init.assert_not_called()
            with self.assertRaises(FileExistsError):
                run(config, full)

    def test_compare_rejects_unmatched_or_incomplete_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = [root / name for name in ("curriculum", "hard")]
            reports = []
            for directory in directories:
                directory.mkdir()
                config = replace(tiny_config(), condition=directory.name)
                target = dict(count=config.final_examples, loss=2.0, accuracy=0.1, baselines={},
                              seed=stream_seed(config.eval_seed, "final", "2", "test"))
                report = dict(step=config.steps, config=config.to_dict(), phase="final",
                              loss_grid={"test": {"2": target}})
                reports.append(report)
                write_json(directory / "final.json", report)
            summary = compare(directories, root / "summary.json")
            self.assertEqual(summary["curriculum_minus_control"]["hard"]["mean"], 0)
            for bad in (dict(step=3), dict(config=replace(config, batch_size=16).to_dict()), dict(phase="monitor")):
                write_json(directories[1] / "final.json", {**reports[1], **bad})
                with self.assertRaises(ValueError):
                    compare(directories, root / "summary.json")


if __name__ == "__main__":
    unittest.main()
