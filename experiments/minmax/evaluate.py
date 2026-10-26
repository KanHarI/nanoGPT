"""Independent procedural banks, answer loss, and structural diagnostics."""

from collections import Counter, defaultdict
import math

import torch
from torch.nn import functional as F

from experiments.finite_field.runtime import autocast
from .data import (DECODE, ENCODE, Generator, IGNORE_INDEX, VOCAB, batch,
                   example_from_ast, parse_rpn, stream_seed)


def prepare(examples, config):
    batches = []
    for offset in range(0, len(examples), config.eval_batch_size):
        chunk = examples[offset:offset + config.eval_batch_size]
        x, y = batch(chunk, config.block_size, pack=config.pack_expressions)
        batches.append((x, y, chunk))
    return batches


@torch.no_grad()
def evaluate_batches(model, batches, config, calibration=None):
    was_training = model.training
    model.eval()
    groups = {name: defaultdict(lambda: [0, 0.0, 0, 0])
              for name in ("root", "operators", "depth", "stack_depth", "negations", "ternary", "alternations", "answer")}
    totals = [0, 0.0, 0, 0]
    frequency_loss = root_loss = 0.0
    minimum_correct = maximum_correct = root_correct = root_count = 0
    try:
        for x, y, examples in batches:
            x, y = x.to(config.device), y.to(config.device)
            with autocast(config):
                logits, _ = model(x, y)
            selected = y != IGNORE_INDEX
            answers = logits[selected].float()
            losses = F.cross_entropy(answers, y[selected], reduction="none").cpu().tolist()
            predictions = answers.argmax(-1).cpu().tolist()
            for example, loss, prediction in zip(examples, losses, predictions):
                token = VOCAB[prediction]
                values = (1, loss, int(token == ENCODE[example.answer]), int(token not in DECODE))
                for index, value in enumerate(values):
                    totals[index] += value
                for name, buckets in groups.items():
                    bucket = buckets[str(getattr(example, name))]
                    for index, value in enumerate(values):
                        bucket[index] += value
                minimum_correct += min(example.leaves) == example.answer
                maximum_correct += max(example.leaves) == example.answer
                if example.root in ("m", "n", "t", "u"):
                    predicted = (min if example.root in ("m", "t") else max)(example.leaves)
                    root_correct += predicted == example.answer
                    root_count += 1
                if calibration:
                    all_counts, root_counts = calibration
                    counts = root_counts.get(example.root, Counter())
                    frequency_loss -= math.log((all_counts[example.answer] + 1) / (sum(all_counts.values()) + 41))
                    root_loss -= math.log((counts[example.answer] + 1) / (sum(counts.values()) + 41))
    finally:
        model.train(was_training)

    def summarize(values):
        count, loss, correct, invalid = values
        return dict(count=count, loss=loss / count, accuracy=correct / count, invalid_rate=invalid / count)

    result = summarize(totals)
    result.update({f"by_{name}": {key: summarize(value) for key, value in sorted(buckets.items())}
                   for name, buckets in groups.items()})
    count = totals[0]
    result["baselines"] = dict(uniform_loss=math.log(41), uniform_accuracy=1 / 41,
                               global_min_accuracy=minimum_correct / count,
                               global_max_accuracy=maximum_correct / count,
                               root_extreme_accuracy=root_correct / root_count if root_count else None,
                               root_extreme_coverage=root_count / count)
    if calibration:
        result["baselines"].update(frequency_loss=frequency_loss / count,
                                    root_frequency_loss=root_loss / count)
    return result


class Evaluation:
    """Both 'train' and 'test' are evaluation streams, never optimizer data."""

    def __init__(self, config, final=False):
        self.phase = "final" if final else "monitor"
        self.count = config.final_examples if final else config.eval_examples
        self.banks, self.seeds, self.baselines = {}, {}, {}
        for gamma in dict.fromkeys([*config.eval_gammas, config.gamma_high]):
            key = f"{gamma:g}"
            sampler = Generator(stream_seed(config.eval_seed, self.phase, key, "calibration"),
                                config.max_operators, config.operator_weights)
            counts, roots = Counter(), defaultdict(Counter)
            for _ in range(config.baseline_examples):
                example = sampler.sample(gamma)
                counts[example.answer] += 1
                roots[example.root][example.answer] += 1
            self.baselines[key] = counts, roots
            for split in ("train", "test"):
                seed = stream_seed(config.eval_seed, self.phase, key, split)
                sampler = Generator(seed, config.max_operators, config.operator_weights)
                examples = [sampler.sample(gamma) for _ in range(self.count)]
                self.banks[split, key] = prepare(examples, config)
                self.seeds[split, key] = seed

    def evaluate(self, model, config):
        grid = {"train": {}, "test": {}}
        for (split, gamma), batches in self.banks.items():
            result = evaluate_batches(model, batches, config, self.baselines[gamma])
            grid[split][gamma] = dict(gamma=float(gamma), seed=self.seeds[split, gamma], **result)
        return grid


def structural_checks(model, config):
    # Paired trees have the same leaves/root/operator multiset but different labels.
    pairs = (("01n23nm", "03n12nm"), ("01m23mn", "03m12mn"))
    results = []
    for pair in pairs:
        examples = [example_from_ast(parse_rpn(expression)) for expression in pair]
        result = evaluate_batches(model, prepare(examples, config), config)
        results.append(dict(examples=[e.text.rstrip() for e in examples], **result))
    return dict(pairs=results, both_correct_fraction=sum(r["accuracy"] == 1 for r in results) / len(results),
                note="Small diagnostic set, not the procedural primary endpoint")


def main():
    import argparse
    from dataclasses import replace
    from experiments.finite_field.runtime import configure
    from .config import Config
    from .train import make_model, write_json

    parser = argparse.ArgumentParser(description="Re-evaluate a trusted min/max checkpoint")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--gammas", nargs="+", type=float)
    parser.add_argument("--examples", type=int)
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = Config(**checkpoint["config"])
    overrides = {"device": args.device, "eval_gammas": args.gammas, "final_examples": args.examples}
    config = replace(config, **{k: v for k, v in overrides.items() if v is not None}).validate()
    configure(config)
    model = make_model(config)
    model.load_state_dict(checkpoint["model"])
    report = dict(step=checkpoint["step"], phase="reevaluation", config=config.to_dict(),
                  training_config=checkpoint["config"], loss_grid=Evaluation(config, final=True).evaluate(model, config),
                  structural_checks=structural_checks(model, config))
    write_json(args.output, report)
    for gamma, result in report["loss_grid"]["test"].items():
        print(f"gamma={gamma}: loss={result['loss']:.4f}, accuracy={result['accuracy']:.3f}")


if __name__ == "__main__":
    main()
