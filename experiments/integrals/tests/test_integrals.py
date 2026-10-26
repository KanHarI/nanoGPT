from dataclasses import replace
from fractions import Fraction
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from experiments.integrals.config import Config, batch_plan, learning_rate
from experiments.integrals.data import (Dataset, Example, FIELD_IDS, IGNORE_INDEX, STOI, VOCAB,
                                       batch, label_from_checks, stream_seed)
from experiments.integrals.encoding import Encoded, MAX_VALUE, MIN_POSITIVE, decode, encode, stable_encoding
from experiments.integrals.evaluate import (Evaluation, evaluate_examples, generated_answers,
                                           loss_totals, summarize_losses)
from experiments.integrals.tracking import grid_metrics
from experiments.integrals.train import run
from experiments.proposals.integral_ast_preview import ARITY
from experiments.proposals.integral_distribution import evaluate
from experiments.proposals.integral_distribution_refine import tree_from_rpn


def example(negations=0):
    value = str((-1)**negations)
    return Example(("0", "1", "1", *(("NEG",)*negations), "INT_X", "="),
                   encode(value).tokens, value, "0", float(negations), negations)


def checks(value=1.0):
    return {"checks":[{"value":value, "estimated_error":1e-14} for _ in range(2)]}


class EncodingTests(unittest.TestCase):
    def test_all_canonical_encodings_and_midpoints(self):
        positive = [Encoded(0, 0, m) for m in range(256)]
        positive += [Encoded(0, e, m) for e in range(1, 32) for m in range(128, 256)]
        count = 0
        for a in positive:
            for sign in (0, 1) if a.mantissa else (0,):
                value = replace(a, sign=sign)
                self.assertEqual(encode(value.value), value)
                self.assertEqual(decode(value.tokens), value)
                count += 1
        self.assertEqual(count, 8447)
        for left, right in zip(positive, positive[1:]):
            midpoint = (left.value+right.value)/2
            expected = left if left.mantissa % 2 == 0 else right
            self.assertEqual(encode(midpoint), expected)
            epsilon = (right.value-left.value)/1000
            self.assertEqual(encode(midpoint-epsilon), left)
            self.assertEqual(encode(midpoint+epsilon), right)

    def test_zero_underflow_range_and_canonical_rejections(self):
        self.assertEqual(encode("-0"), Encoded(0, 0, 0))
        self.assertEqual(encode(MIN_POSITIVE/2), Encoded(0, 0, 0))
        self.assertEqual(encode(-MIN_POSITIVE/2), Encoded(0, 0, 0))
        self.assertEqual(encode(MIN_POSITIVE*Fraction(3, 2)), Encoded(0, 0, 2))
        self.assertEqual(encode(MAX_VALUE), Encoded(0, 31, 255))
        self.assertEqual(encode(-MAX_VALUE), Encoded(1, 31, 255))
        for args in ((1, 0, 0), (0, 1, 0), (0, 1, 127)):
            with self.assertRaises(ValueError):
                Encoded(*args)
        for value in (MAX_VALUE+1, -MAX_VALUE-1, float("inf"), float("nan")):
            with self.assertRaises((ValueError, OverflowError)):
                encode(value)
        bad = list(encode(1).tokens)
        bad[2] = "M0"
        with self.assertRaises(ValueError):
            decode(bad)

    def test_uncertainty_crossing_rounding_or_range_is_rejected(self):
        self.assertEqual(stable_encoding(1, Fraction(1, 10000)), encode(1))
        self.assertEqual(stable_encoding(0, MIN_POSITIVE/4), encode(0))
        with self.assertRaises(ValueError):
            stable_encoding(Fraction(257, 256), Fraction(1, 10000))
        with self.assertRaises(ValueError):
            stable_encoding(MAX_VALUE, Fraction(1, 100))


class DataAndLossTests(unittest.TestCase):
    def test_complete_packing_supervises_every_real_token(self):
        examples = [example(), example(2), example(1)]
        x, y = batch(examples, 50)
        self.assertEqual(x.size(0), 2)
        self.assertTrue((x[:, 0] == STOI["\n"]).all())
        recovered = [VOCAB[t] for t in y[y != IGNORE_INDEX].tolist()]
        self.assertEqual(recovered, [t for e in examples for t in e.tokens])
        self.assertEqual(recovered.count("\n"), 3)
        for i in range(len(x)):
            length = (y[i] != IGNORE_INDEX).sum().item()
            self.assertTrue(torch.equal(x[i, 1:length], y[i, :length-1]))
            self.assertEqual(y[i, length-1], STOI["\n"])
            self.assertTrue((x[i, length:] == STOI["<PAD>"]).all())
        self.assertEqual(batch(examples, 50, pack=False)[0].size(0), 3)
        with self.assertRaises(ValueError):
            batch([example(30)], 22)

    def test_loss_decomposition_counts_padding_and_prompt_gradients(self):
        _, y = batch([example(), example(8)], 128)
        torch.manual_seed(3)
        logits = torch.randn(*y.shape, len(VOCAB), requires_grad=True)
        totals = loss_totals(logits, y)
        metrics = summarize_losses(totals)
        fraction = metrics["answer_token_fraction"]
        self.assertAlmostEqual(metrics["full_nll"], fraction*metrics["answer_nll"]+
                               (1-fraction)*metrics["non_answer_nll"])
        self.assertAlmostEqual(metrics["answer_nll"], (metrics["sign_nll"]+
                               5*metrics["exponent_nll"]+8*metrics["mantissa_nll"])/14)
        self.assertEqual(totals["answer"][1], 28)
        self.assertEqual([totals[n][1] for n in FIELD_IDS], [2, 10, 16])
        full_loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), y.reshape(-1), ignore_index=-1)
        self.assertAlmostEqual(full_loss.item(), metrics["full_nll"], places=6)
        full_loss.backward()
        self.assertGreater(logits.grad[0, 0].abs().sum().item(), 0)  # First bound is supervised.
        self.assertEqual(logits.grad[y == IGNORE_INDEX].abs().sum().item(), 0)

    def test_checked_label_and_cached_target_validation(self):
        value, radius, encoded = label_from_checks(1., checks())
        self.assertEqual(encoded, encode(1))
        self.assertGreater(Fraction(radius), 0)
        bad = checks(float(Fraction(257, 256)))
        with self.assertRaises(ValueError):
            label_from_checks(float(Fraction(257, 256)), bad)
        cached = example().to_dict()
        cached["answer"] = list(encode(2).tokens)
        with self.assertRaises(ValueError):
            Example.from_dict(cached)
        for prefix in ((), ("0", "1", "+", "INT_X", "="),
                       ("1", "0", "x", "INT_X", "="),
                       ("0", "1", "S0", "INT_X", "=")):
            with self.assertRaises(ValueError):
                replace(example(), prefix=prefix)
        with self.assertRaises(ValueError):
            label_from_checks(1., {"checks":checks()["checks"][:1]})
        with self.assertRaises(ValueError):
            label_from_checks(1., {"checks":[{"value":1., "estimated_error":float("nan")}]*2})

    def test_real_numerics_produce_expected_quantized_labels(self):
        for rpn, expected in (("0 1 x INT_X =", Fraction(1, 2)),
                              ("-2 2 x INT_X =", 0),
                              ("-2 2 x FLOOR INT_X =", -2),
                              ("-2 2 x CEIL INT_X =", 2),
                              ("-2 2 x FRAC INT_X =", 2),
                              # Independently checked with 70-digit tanh-sinh quadrature.
                              ("-1 1 x COS SQRT INT_X =", "1.82796940830229599752589086819390986781214")):
            with self.subTest(rpn=rpn):
                a, b = rpn.split()[:2]
                value, method, diagnostics = evaluate(tree_from_rpn(rpn), a, b)
                center, uncertainty, encoded = label_from_checks(value, diagnostics)
                self.assertEqual(encoded, encode(expected))
                self.assertLess(abs(float(center)-float(expected)), 1e-8)

    def test_rejection_preserves_k_bounds_and_cache_avoids_reevaluation(self):
        calls = []
        def evaluate(tree, a, b):
            calls.append((sum(t in ARITY for t in tree.tokens()), a, b))
            if len(calls) == 1:
                raise ValueError("Deliberate failed quadrature")
            return 1., "test", checks()
        with tempfile.TemporaryDirectory() as directory, patch("experiments.integrals.data.evaluate", evaluate):
            dataset = Dataset(directory)
            accepted = dataset.sample(3, 123)
            self.assertEqual(accepted.attempts, 2)
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(dataset.sample(3, 123), accepted)
            self.assertEqual(len(calls), 2)
            saved = json.loads(next(Path(directory).glob("*.json")).read_text())
            self.assertEqual(len(saved["rejected_candidates"]), 1)
            self.assertEqual(dataset.stats["cache_hits"], 1)


class ScriptedModel(nn.Module):
    def __init__(self, tokens, prompt_length):
        super().__init__()
        self.tokens, self.prompt_length, self.prompts = tokens, prompt_length, []

    def forward(self, x):
        self.prompts.append(x.clone())
        logits = torch.full((len(x), 1, len(VOCAB)), -100., device=x.device)
        logits[:, :, STOI[self.tokens[x.size(1)-self.prompt_length]]] = 100.
        return logits, None


class EvaluationTests(unittest.TestCase):
    def test_generation_never_teacher_forces_answer_bits(self):
        e = example()
        predicted = encode(-1).tokens
        model = ScriptedModel(predicted, len(e.prefix)+1)
        result = generated_answers(model, [e], Config())
        self.assertTrue(model.training)
        self.assertEqual(result["exact_answer_accuracy"], 0)
        self.assertEqual(result["valid_answer_rate"], 1)
        self.assertEqual(result["decoded_mae"], 2)
        self.assertEqual(result["tolerance_accuracy"], 0)
        self.assertEqual(model.prompts[0][0].tolist(), [STOI["\n"], *(STOI[t] for t in e.prefix)])
        self.assertEqual(model.prompts[1][0, -1].item(), STOI["S1"])  # Gold sign is S0.
        for i, prompt in enumerate(model.prompts):
            self.assertEqual(prompt[0, len(e.prefix)+1:].tolist(), [STOI[t] for t in predicted[:i]])

    def test_exact_and_invalid_answers(self):
        e = example()
        for tokens, accuracy, valid in ((e.answer, 1, 1), (("0",)*16, 0, 0)):
            model = ScriptedModel(tokens, len(e.prefix)+1).eval()
            result = generated_answers(model, [e], Config())
            self.assertFalse(model.training)
            self.assertEqual(result["exact_answer_accuracy"], accuracy)
            self.assertEqual(result["valid_answer_rate"], valid)
            self.assertEqual(result["decoded_mae"], 0 if valid else None)

    def test_nll_weighting_is_independent_of_eval_batch_size(self):
        # Identical logits avoid changing packed context; only accounting changes.
        class Uniform(nn.Module):
            def forward(self, x, y=None):
                return torch.zeros(len(x), x.size(1) if y is not None else 1, len(VOCAB)), None
        examples = [example(i) for i in (0, 1, 5)]
        a = evaluate_examples(Uniform(), examples, replace(Config(), eval_batch_size=1))
        b = evaluate_examples(Uniform(), examples, replace(Config(), eval_batch_size=3))
        self.assertAlmostEqual(a["full_nll"], math.log(len(VOCAB)), places=6)
        self.assertAlmostEqual(a["full_nll"], b["full_nll"])
        self.assertAlmostEqual(a["answer_token_fraction"], b["answer_token_fraction"])

    def test_banks_are_independent_and_wandb_metric_names(self):
        seeds = []
        class FixtureDataset:
            def examples(self, gamma, seed, count):
                seeds.append(seed)
                return [example()]*count
        config = replace(Config(), eval_examples=1, final_examples=1)
        monitoring = Evaluation(config, FixtureDataset())
        Evaluation(config, FixtureDataset(), final=True)
        self.assertEqual(len(seeds), 120)  # 80 evaluation streams + 20 shared calibration banks read twice.
        self.assertEqual(len(set(seeds)), 100)
        training_seeds = {spec.seed for spec in batch_plan(config)}
        self.assertFalse(training_seeds & set(seeds))
        grid = {"train": {}, "test": {}}
        for split, gamma in monitoring.banks:
            grid[split][gamma] = {name+"_nll":1. for name in
                                  ("answer", "full", "non_answer", "sign", "exponent", "mantissa")}
        logged = grid_metrics(grid, 4)
        self.assertEqual(len(logged), 241)
        self.assertEqual(logged["eval/gamma_20/test/answer_nll"], 1)
        self.assertEqual(logged["optimizer_step"], 4)


class TrainingTests(unittest.TestCase):
    def test_curriculum_shuffled_have_exact_same_batch_identities_and_constant_lr(self):
        c = replace(Config(), steps=10, warmup_steps=2)
        curriculum, shuffled, hard = (batch_plan(replace(c, condition=condition))
                                      for condition in ("curriculum", "shuffled", "hard"))
        self.assertEqual(sorted(curriculum, key=lambda s:s.index), sorted(shuffled, key=lambda s:s.index))
        self.assertNotEqual(curriculum, shuffled)
        self.assertEqual([s.gamma for s in curriculum], [1, 5.75, 10.5, 15.25, 20, 20, 20, 20, 20, 20])
        self.assertTrue(all(s.gamma == 20 for s in hard))
        self.assertEqual([learning_rate(c, i) for i in range(10)], [c.learning_rate/2]+[c.learning_rate]*9)

    def test_resume_matches_uninterrupted_with_dropout_and_full_objective(self):
        def fixture(self, gamma, seed):
            # Labels are exact; sample identities produce different complete expressions.
            return replace(example(seed % 3), gamma=gamma)
        with tempfile.TemporaryDirectory() as directory, patch.object(Dataset, "sample", fixture):
            config = replace(Config(), steps=4, batch_size=3, warmup_steps=1, n_layer=1, n_embd=16,
                             dropout=0.1, eval_gammas=(1, 20), eval_examples=1, final_examples=1,
                             generation_examples=1, eval_interval=2, log_interval=1, threads=1,
                             cache_dir=str(Path(directory)/"cache"))
            full = run(config, Path(directory)/"full")
            run(config, Path(directory)/"resumed", stop_after=2)
            resumed = run(config, Path(directory)/"resumed", resume=True)
            a = torch.load(full, weights_only=False)
            b = torch.load(resumed, weights_only=False)
            progress = json.loads((resumed.parent/"checkpoint_progress.json").read_text())
            self.assertEqual(progress, {k:b[k] for k in ("step", "records", "history_records", "completed")})
            for key in a["model"]:
                self.assertTrue(torch.equal(a["model"][key], b["model"][key]), key)
            self.assertTrue(torch.equal(a["torch_rng"], b["torch_rng"]))
            self.assertEqual(a["supervised_tokens"], b["supervised_tokens"])
            records = [json.loads(row) for row in (full.parent/"metrics.jsonl").read_text().splitlines()]
            for row in records:
                if row["kind"] == "train":
                    self.assertGreater(row["grad_norm"], 0)
                    self.assertTrue(math.isfinite(row["full_nll"]))


if __name__ == "__main__":
    unittest.main()
