"""Balanced, answer-only evaluation with independent label-frequency baselines."""

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from .config import Config
from .data import Generator, IGNORE_INDEX, NUMERALS, PRIMES, VOCAB, batch
from .runtime import autocast, configure


class EvaluationSet:
    """A replayable procedural sample. Evaluation never touches training RNGs."""

    def __init__(self, config: Config, gamma: float, per_field: int, seed: int):
        self.gamma, self.seed = gamma, seed
        sampler = Generator(seed, config.max_operators, config.include_power,
                            include_division=config.include_division)
        self.examples = [sampler.sample(gamma, prime)
                         for _ in range(per_field) for prime in PRIMES]
        # Fit the constant-per-field predictor on separate procedural examples.
        calibration = Generator(seed + 1_000_003, config.max_operators, config.include_power,
                                include_division=config.include_division)
        counts = {p: Counter(calibration.sample(gamma, p).answer
                             for _ in range(config.baseline_per_field)) for p in PRIMES}
        self.probs = {p: [(counts[p][a] + 1) / (config.baseline_per_field + p)
                         for a in range(p)] for p in PRIMES}
        self.majority = {p: max(range(p), key=lambda a: counts[p][a]) for p in PRIMES}
        self.generation_attempts = sampler.attempts

    @torch.no_grad()
    def evaluate(self, model, config: Config) -> dict:
        was_training = model.training
        model.eval()
        groups = {name: defaultdict(lambda: [0, 0.0, 0, 0, 0.0, 0])
                  for name in ("field", "operators", "depth")}
        try:
            batch_size = config.eval_batch_size or config.batch_size
            for offset in range(0, len(self.examples), batch_size):
                examples = self.examples[offset:offset + batch_size]
                x, y = batch(examples, config.block_size, config.device, pack=config.pack_expressions)
                with autocast(config):
                    logits, _ = model(x, y)
                selected = y != IGNORE_INDEX
                answer_logits, labels = logits[selected].float(), y[selected]
                losses = F.cross_entropy(answer_logits, labels, reduction="none").cpu().tolist()
                predictions = answer_logits.argmax(dim=-1).cpu().tolist()
                for example, loss, prediction in zip(examples, losses, predictions):
                    predicted = VOCAB[prediction]
                    correct = int(predicted == NUMERALS[example.answer])
                    invalid = int(predicted not in NUMERALS[:example.prime])
                    baseline_loss = -math.log(self.probs[example.prime][example.answer])
                    baseline_correct = int(self.majority[example.prime] == example.answer)
                    values = (1, loss, correct, invalid, baseline_loss, baseline_correct)
                    for name, key in (("field", example.prime), ("operators", example.operators),
                                      ("depth", example.depth)):
                        aggregate = groups[name][key]
                        for i, value in enumerate(values):
                            aggregate[i] += value
        finally:
            model.train(was_training)

        def summarize(values):
            count, loss, correct, invalid, baseline_loss, baseline_correct = values
            return dict(count=count, loss=loss / count, accuracy=correct / count,
                        invalid_rate=invalid / count, frequency_baseline_loss=baseline_loss / count,
                        frequency_baseline_accuracy=baseline_correct / count)

        totals = [sum(values[i] for values in groups["field"].values()) for i in range(6)]
        return dict(gamma=self.gamma, seed=self.seed, **summarize(totals),
                    by_field={str(k): summarize(v) for k, v in sorted(groups["field"].items())},
                    by_operators={str(k): summarize(v) for k, v in sorted(groups["operators"].items())},
                    by_depth={str(k): summarize(v) for k, v in sorted(groups["depth"].items())},
                    generation_attempts=self.generation_attempts)


class LossGrid:
    """Independent procedural train/test banks, balanced at every gamma/field.

    Both names denote generator streams, not seen/unseen finite-corpus splits.
    Neither bank is used by the optimizer. Complete expressions are packed using
    the training serialization and kept fixed for repeatable monitored curves.
    """

    def __init__(self, config, per_field=None, seed=None):
        self.per_field = config.eval_per_field if per_field is None else per_field
        self.seed = config.eval_seed if seed is None else seed
        self.banks = {}
        batch_size = config.eval_batch_size or config.batch_size
        for split in ("train", "test"):
            seed = self.seed + (30_000_041 if split == "train" else 0)
            for gamma in dict.fromkeys([*config.eval_gammas, config.gamma_high]):
                sampler = Generator(seed, config.max_operators, config.include_power,
                                    include_division=config.include_division)
                examples = [sampler.sample(gamma, prime) for _ in range(self.per_field) for prime in PRIMES]
                batches = []
                for offset in range(0, len(examples), batch_size):
                    chunk = examples[offset:offset + batch_size]
                    x, y = batch(chunk, config.block_size, pack=config.pack_expressions)
                    fields = [example.prime for example in chunk]
                    batches.append((x, y, fields))
                self.banks[split, gamma] = batches

    @torch.no_grad()
    def evaluate(self, model, config):
        was_training = model.training
        model.eval()
        results = {"train": {}, "test": {}}
        try:
            for (split, gamma), batches in self.banks.items():
                totals = {prime: 0.0 for prime in PRIMES}
                for x, y, fields in batches:
                    x, y = x.to(config.device), y.to(config.device)
                    with autocast(config):
                        logits, _ = model(x, y)
                    selected = y != IGNORE_INDEX
                    losses = F.cross_entropy(logits[selected].float(), y[selected], reduction="none").cpu().tolist()
                    for prime, loss in zip(fields, losses):
                        totals[prime] += loss
                results[split][f"{gamma:g}"] = dict(
                    gamma=gamma, seed=self.seed + (30_000_041 if split == "train" else 0),
                    count=self.per_field * len(PRIMES),
                    loss=sum(totals.values()) / (self.per_field * len(PRIMES)),
                    by_field={str(p): dict(count=self.per_field, loss=totals[p] / self.per_field) for p in PRIMES})
        finally:
            model.train(was_training)
        return results


def evaluate_checkpoint(checkpoint_path, output_path=None, gammas=None, per_field=None, device=None):
    from model import GPT, GPTConfig

    # Checkpoints contain Python RNG/optimizer state. Load only trusted artifacts.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = Config.from_saved(checkpoint["config"])
    if device is not None:
        config.device = device
        config.validate()
    configure(config)
    model = GPT(GPTConfig(**checkpoint["model_args"]))
    model.load_state_dict(checkpoint["model"])
    model.to(config.device)
    gammas = list(dict.fromkeys([config.gamma_high, *(config.eval_gammas if gammas is None else gammas)]))
    if any(not math.isfinite(gamma) or gamma < 0 for gamma in gammas):
        raise ValueError("Evaluation gammas must be finite and nonnegative")
    per_field = config.final_per_field if per_field is None else per_field
    if per_field < 1:
        raise ValueError("per_field must be positive")
    # A separate stream from the repeatedly monitored evaluation set.
    seed = config.eval_seed + 10_000_019
    if config.eval_mode == "loss_grid":
        grid = LossGrid(replace(config, eval_gammas=gammas), per_field, seed).evaluate(model, config)
        report = dict(step=checkpoint["step"], config=config.to_dict(),
                      evaluations=list(grid["test"].values()), loss_grid=grid)
    else:
        results = [EvaluationSet(config, gamma, per_field, seed).evaluate(model, config) for gamma in gammas]
        report = dict(step=checkpoint["step"], config=config.to_dict(), evaluations=results)
    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--gammas", type=float, nargs="+")
    parser.add_argument("--per-field", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    report = evaluate_checkpoint(args.checkpoint, args.output, args.gammas, args.per_field, args.device)
    for result in report["evaluations"]:
        print(f"gamma={result['gamma']:g} loss={result['loss']:.4f}")


if __name__ == "__main__":
    main()
