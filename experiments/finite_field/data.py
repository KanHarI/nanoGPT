"""Pure-Python arithmetic, AST sampling, and the character-level wire format.

The accepted distribution is defined by rejection sampling at fixed (field, K).
It is NOT uniform over tree shapes, nor over valid expressions of all sizes.
"""

from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
import math
import random

NUMERALS = "0123456789abcdefghij"
PRIMES = (2, 3, 5, 7, 11, 13, 17, 19)
# Keep vocabulary identical across operator toggles. Newlines also pad inputs.
VOCAB = NUMERALS + ".+-*/^=\n"
STOI = {char: index for index, char in enumerate(VOCAB)}
IGNORE_INDEX = -1


def operate(op: str, left: int, right: int, prime: int) -> int:
    """All operands/results are canonical residues; ^ uses an integer exponent."""
    if op == "+":
        return (left + right) % prime
    if op == "-":
        return (left - right) % prime
    if op == "*":
        return left * right % prime
    if op == "/":
        if right == 0:
            raise ZeroDivisionError("division by zero in a finite field")
        return left * pow(right, -1, prime) % prime
    if op == "^":
        return pow(left, right, prime)  # Python's modular pow defines 0^0 = 1.
    raise ValueError(f"Unknown operator: {op!r}")


@dataclass(frozen=True)
class Node:
    token: str
    left: "Node | None" = None
    right: "Node | None" = None

    def render(self) -> str:
        if self.left is None:
            return self.token
        return self.left.render() + self.right.render() + self.token

    def evaluate(self, prime: int) -> int:
        if self.left is None:
            return NUMERALS.index(self.token)
        return operate(self.token, self.left.evaluate(prime), self.right.evaluate(prime), prime)

    @property
    def depth(self) -> int:
        return 0 if self.left is None else 1 + max(self.left.depth, self.right.depth)


@dataclass(frozen=True)
class Example:
    prime: int
    expression: str
    answer: int
    operators: int
    depth: int

    @property
    def text(self) -> str:
        return f"{NUMERALS[self.prime]}.{self.expression}={NUMERALS[self.answer]}\n"


def evaluate_rpn(expression: str, prime: int) -> int:
    """Independent stack interpreter used to check AST rendering/evaluation."""
    if prime not in PRIMES:
        raise ValueError("Unsupported prime")
    stack = []
    for token in expression:
        if token in NUMERALS:
            value = NUMERALS.index(token)
            if value >= prime:
                raise ValueError("Leaf outside the selected field")
            stack.append(value)
            continue
        if token not in "+-*/^" or len(stack) < 2:
            raise ValueError("Malformed postfix expression")
        right, left = stack.pop(), stack.pop()
        # Deliberately do not call operate(): a separate arithmetic oracle.
        if token == "+":
            value = left + right
        elif token == "-":
            value = left - right
        elif token == "*":
            value = left * right
        elif token == "/":
            if not right:
                raise ZeroDivisionError("division by zero")
            inverse = next(x for x in range(1, prime) if x * right % prime == 1)
            value = left * inverse
        else:
            value = 1
            for _ in range(right):
                value = value * left % prime
        stack.append(value % prime)
    if len(stack) != 1:
        raise ValueError("Malformed postfix expression")
    return stack[0]


@lru_cache(maxsize=512)
def size_weights(gamma: float, maximum: int) -> tuple[float, ...]:
    """Unnormalized Poisson weights conditioned on 0 <= K <= maximum."""
    if not math.isfinite(gamma) or gamma < 0 or maximum < 0:
        raise ValueError("gamma must be finite/nonnegative and maximum nonnegative")
    if gamma == 0:
        return (1.0,) + (0.0,) * maximum
    logs = [k * math.log(gamma) - math.lgamma(k + 1) for k in range(maximum + 1)]
    peak = max(logs)
    return tuple(math.exp(value - peak) for value in logs)


class Generator:
    def __init__(self, seed: int, max_operators: int = 32, include_power: bool = False,
                 include_division: bool = False):
        if not isinstance(max_operators, int) or not 0 <= max_operators <= 128:
            raise ValueError("max_operators must be an integer between 0 and 128")
        if not isinstance(include_power, bool) or not isinstance(include_division, bool):
            raise ValueError("Operator toggles must be booleans")
        self.rng = random.Random(seed)
        self.max_operators = max_operators
        self.ops = "+-*" + ("/" if include_division else "") + ("^" if include_power else "")
        self.attempts = 0
        self.items = 0
        self.characters = 0
        self.operator_counts = Counter()
        self.field_counts = Counter()

    def _tree(self, prime: int, operators: int) -> Node:
        if operators == 0:
            return Node(NUMERALS[self.rng.randrange(prime)])
        left_size = self.rng.randrange(operators)
        op = self.rng.choice(self.ops)
        return Node(op, self._tree(prime, left_size),
                    self._tree(prime, operators - 1 - left_size))

    def sample(self, gamma: float, prime: int | None = None) -> Example:
        if prime is None:
            prime = self.rng.choice(PRIMES)
        if prime not in PRIMES:
            raise ValueError("Unsupported prime")
        operators = self.rng.choices(range(self.max_operators + 1),
                                     weights=size_weights(gamma, self.max_operators))[0]
        # Retry only the tree, NEVER the field or size. Global rejection would
        # under-sample small fields and large trees, where zero division is common.
        for _ in range(100_000):
            self.attempts += 1
            tree = self._tree(prime, operators)
            try:
                answer = tree.evaluate(prime)
            except ZeroDivisionError:
                continue
            example = Example(prime, tree.render(), answer, operators, tree.depth)
            self.items += 1
            self.characters += len(example.text)
            self.operator_counts.update(c for c in example.expression if c in self.ops)
            self.field_counts[prime] += 1
            return example
        raise RuntimeError("Too many invalid ASTs; reduce complexity or inspect the sampler")

    def state_dict(self) -> dict:
        return dict(ops=self.ops, rng=self.rng.getstate(), attempts=self.attempts, items=self.items,
                    characters=self.characters, operator_counts=dict(self.operator_counts),
                    field_counts=dict(self.field_counts))

    def load_state_dict(self, state: dict) -> None:
        if "ops" in state and state["ops"] != self.ops:
            raise ValueError("Cannot restore sampler state with different operator toggles")
        self.rng.setstate(state["rng"])
        for name in ("attempts", "items", "characters"):
            setattr(self, name, state[name])
        self.operator_counts = Counter(state["operator_counts"])
        self.field_counts = Counter(state["field_counts"])


def batch(examples: list[Example], block_size: int, device="cpu", pack=False):
    """Right-pad whole expressions; optionally pack complete newline-ended lines.

    Future padding and the answer itself cannot influence logits at '=' because
    GPT uses causal attention. No attention-mask modification is necessary.
    """
    import torch

    if not examples:
        raise ValueError("Cannot batch zero examples")
    rows, targets = [], []
    current, current_targets = [], []
    for example in examples:
        tokens = [STOI[c] for c in example.text]
        if len(tokens) - 1 > block_size:
            raise ValueError("Complete expression does not fit the context")
        # The final newline is a next-token target and need not be in x. Every
        # preceding packed item's newline remains in x as an explicit delimiter.
        if current and (not pack or len(current) + len(tokens) - 1 > block_size):
            rows.append(current[:-1])
            targets.append(current_targets[:-1])
            current, current_targets = [], []
        labels = [IGNORE_INDEX] * len(tokens)
        labels[example.text.index("=")] = tokens[example.text.index("=") + 1]
        current.extend(tokens)
        current_targets.extend(labels)
    if current:
        rows.append(current[:-1])
        targets.append(current_targets[:-1])
    # Construct whole tensors once, avoiding one Torch allocation per expression.
    x = torch.tensor([row + [STOI["\n"]] * (block_size - len(row)) for row in rows], dtype=torch.long)
    y = torch.tensor([row + [IGNORE_INDEX] * (block_size - len(row)) for row in targets], dtype=torch.long)
    return x.to(device), y.to(device)
