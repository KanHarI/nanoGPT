"""Single-character integers, unrestricted AST generation, and complete-line packing."""

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import random

from experiments.finite_field.data import size_weights

VERSION = "minmax-v1"
NUMERALS = "0123456789abcdefghijkABCDEFGHIJKLMNOPQRST"
VALUES = tuple(range(21)) + tuple(range(-1, -21, -1))
DECODE = dict(zip(NUMERALS, VALUES))
ENCODE = dict(zip(VALUES, NUMERALS))
ARITY = {"m": 2, "n": 2, "t": 3, "u": 3, "~": 1}
OPS = tuple(ARITY)
VOCAB = NUMERALS + "".join(OPS) + "=\n"
STOI = {token: index for index, token in enumerate(VOCAB)}
IGNORE_INDEX = -1


def stream_seed(seed, *identity):
    """Stable across processes; never use Python's randomized hash()."""
    key = ":".join(map(str, (VERSION, seed, *identity)))
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Node:
    token: str
    children: tuple = ()

    def __post_init__(self):
        if self.token not in DECODE and self.token not in ARITY:
            raise ValueError("Unknown AST token")
        if len(self.children) != ARITY.get(self.token, 0):
            raise ValueError("Wrong AST arity")

    def render(self):
        return "".join(child.render() for child in self.children) + self.token

    def evaluate(self):
        if not self.children:
            return DECODE[self.token]
        values = [child.evaluate() for child in self.children]
        if self.token == "~":
            return -values[0]
        return min(values) if self.token in ("m", "t") else max(values)


def parse_rpn(expression):
    stack = []
    for token in expression:
        if token in DECODE:
            stack.append(Node(token))
        elif token in ARITY and len(stack) >= ARITY[token]:
            arity = ARITY[token]
            children = tuple(stack[-arity:])
            del stack[-arity:]
            stack.append(Node(token, children))
        else:
            raise ValueError("Malformed RPN expression")
    if len(stack) != 1:
        raise ValueError("RPN must leave exactly one value")
    return stack[0]


def evaluate_rpn(expression):
    """Independent numerical stack oracle; does not call Node.evaluate()."""
    stack = []
    for token in expression:
        if token in DECODE:
            stack.append(DECODE[token])
        elif token in ARITY and len(stack) >= ARITY[token]:
            values = [stack.pop() for _ in range(ARITY[token])]
            if token == "~":
                result = -values[0]
            else:
                result = sorted(values)[0 if token in ("m", "t") else -1]
            stack.append(result)
        else:
            raise ValueError("Malformed RPN expression")
    if len(stack) != 1:
        raise ValueError("RPN must leave exactly one value")
    return stack[0]


@dataclass(frozen=True)
class Example:
    expression: str
    answer: int
    operators: int
    depth: int
    stack_depth: int
    negations: int
    ternary: int
    alternations: int
    leaves: tuple

    @property
    def text(self):
        return f"{self.expression}={ENCODE[self.answer]}\n"

    @property
    def root(self):
        return self.expression[-1] if self.operators else "leaf"


def example_from_ast(tree):
    expression = tree.render()
    counts = Counter(expression)
    stack, maximum = [], 0
    # Track structural min/max alternations, ignoring negations, along paths.
    def walk(node, previous=None, changes=0):
        kind = ("min" if node.token in ("m", "t") else
                "max" if node.token in ("n", "u") else previous)
        changes += int(previous is not None and kind != previous)
        return max([changes, *(walk(child, kind, changes) for child in node.children)])
    for token in expression:
        if token in DECODE:
            stack.append(0)
            maximum = max(maximum, len(stack))
        else:
            depths = stack[-ARITY[token]:]
            del stack[-ARITY[token]:]
            stack.append(1 + max(depths))
    return Example(expression, tree.evaluate(), sum(counts[op] for op in OPS),
                   stack[0], maximum, counts["~"], counts["t"] + counts["u"], walk(tree),
                   tuple(DECODE[token] for token in expression if token in DECODE))


def split_counts(total, arity, rng):
    """Uniform ordered weak composition using uniformly sampled stars-and-bars."""
    bars = sorted(rng.sample(range(total + arity - 1), arity - 1))
    bounds = [-1, *bars, total + arity - 1]
    return tuple(right - left - 1 for left, right in zip(bounds, bounds[1:]))


class Generator:
    def __init__(self, seed, max_operators=32, operator_weights=(1, 1, 1, 1, 1)):
        import math
        if type(max_operators) is not int or not 0 <= max_operators <= 128:
            raise ValueError("max_operators must be an integer in [0, 128]")
        if (len(operator_weights) != len(OPS) or
                any(not math.isfinite(w) or w < 0 for w in operator_weights) or
                sum(operator_weights) <= 0):
            raise ValueError("Provide five finite nonnegative operator weights with positive sum")
        self.rng = random.Random(seed)
        self.maximum, self.weights = max_operators, operator_weights

    def tree(self, operators):
        if operators == 0:
            return Node(self.rng.choice(NUMERALS))
        token = self.rng.choices(OPS, weights=self.weights)[0]
        sizes = split_counts(operators - 1, ARITY[token], self.rng)
        return Node(token, tuple(self.tree(size) for size in sizes))

    def sample(self, gamma):
        count = self.rng.choices(range(self.maximum + 1), weights=size_weights(gamma, self.maximum))[0]
        return example_from_ast(self.tree(count))


def batch(examples, block_size, device="cpu", pack=True):
    """Pack whole lines; '=' predicts the answer, all other targets are masked."""
    import torch
    if not examples:
        raise ValueError("Empty batch")
    rows, labels, current, targets = [], [], [], []
    for example in examples:
        text = example.text
        tokens = [STOI[token] for token in text]
        if len(tokens) - 1 > block_size:
            raise ValueError("Complete expression does not fit context")
        if current and (not pack or len(current) + len(tokens) - 1 > block_size):
            rows.append(current[:-1])
            labels.append(targets[:-1])
            current, targets = [], []
        answer_targets = [IGNORE_INDEX] * len(tokens)
        answer_targets[text.index("=")] = STOI[ENCODE[example.answer]]
        current.extend(tokens)
        targets.extend(answer_targets)
    rows.append(current[:-1])
    labels.append(targets[:-1])
    x = torch.tensor([row + [STOI["\n"]] * (block_size - len(row)) for row in rows], dtype=torch.long)
    y = torch.tensor([row + [IGNORE_INDEX] * (block_size - len(row)) for row in labels], dtype=torch.long)
    return x.to(device), y.to(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gamma", type=float, default=12)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sampler = Generator(args.seed)
    for _ in range(args.count):
        print(sampler.sample(args.gamma).text, end="")


if __name__ == "__main__":
    main()
