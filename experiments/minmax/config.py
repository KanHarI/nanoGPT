"""Model defaults and exact-batch curriculum controls."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

from .data import Generator, VERSION, stream_seed

CONDITIONS = ("curriculum", "hard", "shuffled", "shuffled_ramp")


@dataclass
class Config:
    version: str = VERSION
    condition: str = "curriculum"
    seed: int = 0
    gamma_low: float = 1
    gamma_high: float = 12
    ramp_fraction: float = 0.4
    max_operators: int = 32
    operator_weights: tuple = (1, 1, 1, 1, 1)
    steps: int = 2000
    batch_size: int = 64
    block_size: int = 128
    pack_expressions: bool = True
    n_layer: int = 24
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    learning_rate: float = 0.0006
    warmup_steps: int = 40
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_examples: int = 128
    final_examples: int = 512
    baseline_examples: int = 1024
    eval_batch_size: int = 128
    eval_seed: int = 20261017
    eval_gammas: tuple = tuple(range(1, 13))
    log_interval: int = 20
    threads: int = 4
    device: str = "cpu"
    precision: str = "float32"

    def validate(self):
        if self.version != VERSION or self.condition not in CONDITIONS:
            raise ValueError("Unsupported generator version or condition")
        for name in ("steps", "batch_size", "block_size", "n_layer", "n_head", "n_embd",
                     "eval_interval", "eval_examples", "final_examples", "baseline_examples",
                     "eval_batch_size", "log_interval", "threads"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or type(self.eval_seed) is not int:
            raise ValueError("Seeds must be integers")
        Generator(0, self.max_operators, self.operator_weights)
        if self.block_size < 3 * self.max_operators + 3:
            raise ValueError("Context must fit every full expression: block_size >= 3*max_operators+3")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding width must be divisible by heads")
        if not 0 < self.ramp_fraction <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid ramp fraction or dropout")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("Warmup must be nonnegative and shorter than training")
        for name in ("learning_rate", "weight_decay", "grad_clip"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0 or (name != "weight_decay" and value == 0):
                raise ValueError(f"Invalid {name}")
        if not self.eval_gammas or len(set(self.eval_gammas)) != len(self.eval_gammas):
            raise ValueError("Evaluation gammas must be nonempty and unique")
        if any(not math.isfinite(gamma) or gamma < 0
               for gamma in (self.gamma_low, self.gamma_high, *self.eval_gammas)):
            raise ValueError("Gammas must be finite and nonnegative")
        if self.gamma_low > self.gamma_high:
            raise ValueError("gamma_low exceeds gamma_high")
        if type(self.pack_expressions) is not bool:
            raise ValueError("pack_expressions must be boolean")
        if self.device not in ("cpu", "cuda", "mps") or self.precision not in ("float32", "bfloat16"):
            raise ValueError("Invalid device or precision")
        if self.device == "mps" and self.precision != "float32":
            raise ValueError("Use float32 on MPS")
        return self

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()


@dataclass(frozen=True)
class BatchSpec:
    index: int
    gamma: float
    seed: int


def batch_plan(config):
    ramp = max(2, math.ceil(config.steps * config.ramp_fraction))
    indices = list(range(config.steps))
    rng = random.Random(stream_seed(config.seed, "shuffle"))
    if config.condition == "shuffled":
        rng.shuffle(indices)
    elif config.condition == "shuffled_ramp":
        prefix = indices[:ramp]
        rng.shuffle(prefix)
        indices[:ramp] = prefix
    return [BatchSpec(index,
                      config.gamma_high if config.condition == "hard" else
                      config.gamma_low + (config.gamma_high - config.gamma_low) * min(index / (ramp - 1), 1),
                      stream_seed(config.seed, "training-batch", index)) for index in indices]


def generate_batch(config, spec):
    sampler = Generator(spec.seed, config.max_operators, config.operator_weights)
    return [sampler.sample(spec.gamma) for _ in range(config.batch_size)]


def learning_rate(config, step):
    return config.learning_rate * min((step + 1) / max(config.warmup_steps, 1), 1)
