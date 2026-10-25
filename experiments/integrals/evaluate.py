"""Token-weighted NLL breakdown and unprompted, unconstrained answer generation."""

from collections import defaultdict
import math

import torch
from torch.nn import functional as F

from experiments.finite_field.runtime import autocast
from .data import FIELD_IDS, IGNORE_INDEX, STOI, VOCAB, batch, stream_seed
from .encoding import ANSWER_LENGTH, decode, encode, rational

BIT_POSITIONS = (0, *range(2,7), *range(8,16))


def fit_bit_prior(examples):
    """Jeffreys-smoothed per-position bit frequencies from an independent bank."""
    return [(sum(e.answer[position].endswith("1") for e in examples)+.5)/(len(examples)+1)
            for position in BIT_POSITIONS]


def baseline_scores(examples, probabilities, config):
    loss = sum(-math.log(p if e.answer[position].endswith("1") else 1-p)
               for e in examples for position,p in zip(BIT_POSITIONS,probabilities))
    zero = encode(0).tokens
    return dict(answer_bit_prior_nll=loss/(14*len(examples)),
                always_zero_exact_accuracy=sum(e.answer == zero for e in examples)/len(examples),
                always_zero_tolerance_accuracy=sum(abs(float(e.value)) <= config.absolute_tolerance+
                                                     config.relative_tolerance*abs(float(e.value))
                                                     for e in examples)/len(examples))


def loss_totals(logits, targets):
    """Return sums/counts, so merging unequal-length batches is unbiased."""
    valid = targets != IGNORE_INDEX
    losses = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), targets.reshape(-1),
                             ignore_index=IGNORE_INDEX, reduction="none").reshape_as(targets)
    masks = {name: (targets == ids[0]) | (targets == ids[1]) for name, ids in FIELD_IDS.items()}
    masks["answer"] = masks["sign"] | masks["exponent"] | masks["mantissa"]
    masks["full"] = valid
    masks["non_answer"] = valid & ~masks["answer"]
    return {name: (losses[mask].double().sum().item(), mask.sum().item()) for name, mask in masks.items()}


def summarize_losses(totals):
    result = {name+"_nll": total/count if count else None for name, (total, count) in totals.items()}
    result["answer_token_fraction"] = totals["answer"][1]/totals["full"][1]
    result["tokens"] = totals["full"][1]
    result["answer_bits"] = totals["answer"][1]
    return result


@torch.no_grad()
def predict_answers(model, examples, config):
    """Generate from prompts only, batching compatible lengths across all banks."""
    groups = defaultdict(list)
    for index, example in enumerate(examples):
        groups[len(example.prefix)].append((index, example))
    predictions = [None]*len(examples)
    was_training = model.training
    model.eval()
    try:
        for group in groups.values():
            for offset in range(0, len(group), config.eval_batch_size):
                indexed = group[offset:offset+config.eval_batch_size]
                chunk = [example for _,example in indexed]
                prompt = torch.tensor([[STOI["\n"], *(STOI[t] for t in e.prefix)] for e in chunk],
                                      device=config.device, dtype=torch.long)
                prompt_length = prompt.size(1)
                if prompt_length+ANSWER_LENGTH-1 > config.block_size:
                    raise ValueError("Free generation would truncate the expression")
                for _ in range(ANSWER_LENGTH):
                    with autocast(config):
                        logits, _ = model(prompt)
                    prompt = torch.cat((prompt, logits[:, -1, :].argmax(-1, keepdim=True)), dim=1)
                for (index, _), ids in zip(indexed, prompt[:, prompt_length:].cpu().tolist()):
                    predictions[index] = tuple(VOCAB[i] for i in ids)
    finally:
        model.train(was_training)
    return predictions


def score_predictions(examples, predictions, config):
    valid = correct = within = 0
    error_sum = 0.0
    if len(examples) != len(predictions):
        raise ValueError("Prediction count mismatch")
    for example, tokens in zip(examples, predictions):
        correct += tokens == example.answer
        try:
            prediction = decode(tokens).value
        except ValueError:
            continue
        valid += 1
        reference = rational(example.value)
        error = abs(prediction-reference)
        error_sum += float(error)
        within += error <= rational(config.absolute_tolerance)+rational(config.relative_tolerance)*abs(reference)
    count = len(examples)
    if not count:
        raise ValueError("Free generation requires at least one example")
    return dict(generation_examples=count, valid_answers=valid, valid_answer_rate=valid/count,
                exact_answer_accuracy=correct/count, tolerance_accuracy=within/count,
                decoded_mae=error_sum/valid if valid else None)


def generated_answers(model, examples, config):
    return score_predictions(examples, predict_answers(model, examples, config), config)


@torch.no_grad()
def evaluate_examples(model, examples, config, include_generation=True):
    was_training = model.training
    model.eval()
    totals = defaultdict(lambda: [0.0, 0])
    slices = defaultdict(lambda:[0.0,0])
    try:
        for offset in range(0, len(examples), config.eval_batch_size):
            chunk = examples[offset:offset+config.eval_batch_size]
            x, y = batch(chunk, config.block_size,
                         config.device, config.pack_expressions)
            with autocast(config):
                logits, _ = model(x, y)
            for name, (loss, count) in loss_totals(logits, y).items():
                totals[name][0] += loss
                totals[name][1] += count
            answer_mask = torch.zeros_like(y, dtype=torch.bool)
            for ids in FIELD_IDS.values():
                answer_mask |= (y == ids[0]) | (y == ids[1])
            per_example = F.cross_entropy(logits[answer_mask].float(), y[answer_mask], reduction="none").reshape(-1,14).mean(-1).tolist()
            for example, loss in zip(chunk, per_example):
                names = ("with_x" if "x" in example.prefix else "without_x",
                         "quantized_zero" if example.answer == encode(0).tokens else "nonzero",
                         "with_rounding" if set(example.prefix) & {"FLOOR", "CEIL", "FRAC"} else "without_rounding")
                for name in names:
                    slices[name][0] += loss
                    slices[name][1] += 1
        result = dict(examples=len(examples), **summarize_losses(totals))
        result["slices"] = {name:dict(answer_nll=total/count, examples=count) for name,(total,count) in slices.items()}
        if include_generation:
            result.update(generated_answers(model, examples[:config.generation_examples], config))
        return result
    finally:
        model.train(was_training)


class Evaluation:
    """Independent procedural train/test monitoring streams, never optimizer data."""

    def __init__(self, config, dataset, final=False):
        phase = "final" if final else "monitor"
        self.generation_count = (config.final_generation_examples or config.generation_examples) if final else config.generation_examples
        count = config.final_examples if final else config.eval_examples
        self.banks = {}
        self.baselines = {}
        for gamma in dict.fromkeys([*config.eval_gammas, config.gamma_high]):
            key = f"{gamma:g}"
            calibration = dataset.examples(gamma, stream_seed(config.eval_seed, "baseline", key), config.baseline_examples)
            self.baselines[key] = fit_bit_prior(calibration)
            for split in ("train", "test"):
                key = f"{gamma:g}"
                seed = stream_seed(config.eval_seed, phase, key, split)
                self.banks[split, key] = (seed, dataset.examples(gamma, seed, count))

    def evaluate(self, model, config, include_generation=True):
        grid = {"train": {}, "test": {}}
        flat, spans = [], []
        for (split, gamma), (seed, examples) in self.banks.items():
            grid[split][gamma] = dict(gamma=float(gamma), seed=seed,
                                     **evaluate_examples(model, examples, config, include_generation=False),
                                     **baseline_scores(examples, self.baselines[gamma], config))
            selected = examples[:self.generation_count]
            spans.append((split, gamma, len(flat), len(selected)))
            flat.extend(selected)
        if include_generation:
            predictions = predict_answers(model, flat, config)
            for split, gamma, first, count in spans:
                grid[split][gamma].update(score_predictions(flat[first:first+count],
                                                          predictions[first:first+count], config))
        return grid
