"""Integral curriculum, exact-batch shuffled control, and constant-after-warmup LR."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random

from .data import VERSION, stream_seed

CONDITIONS = ("curriculum", "hard", "shuffled")


@dataclass
class Config:
    version: str = VERSION
    condition: str = "curriculum"
    seed: int = 0
    gamma_low: float = 1
    gamma_high: float = 20
    ramp_fraction: float = 0.5
    ramp_steps: int = 0  # Explicit calibrated override; zero uses ramp_fraction.
    max_operators: int = 32
    steps: int = 2000
    batch_size: int = 64  # Examples per optimizer update, before packing.
    block_size: int = 128
    pack_expressions: bool = True
    n_layer: int = 24
    n_head: int = 4
    n_embd: int = 256
    dropout: float = 0.0
    learning_rate: float = 0.0003
    warmup_steps: int = 40
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_examples: int = 128
    final_examples: int = 512
    baseline_examples: int = 128
    eval_batch_size: int = 128
    generation_examples: int = 16  # Per split/gamma, from that evaluation bank.
    final_generation_examples: int = 0  # Zero reuses generation_examples.
    generation_interval: int = 0  # Zero decodes at every NLL evaluation.
    checkpoint_interval: int = 0  # Zero checkpoints at every NLL evaluation.
    eval_seed: int = 20261024
    eval_gammas: tuple = tuple(range(1, 21))
    absolute_tolerance: float = 1e-6
    relative_tolerance: float = 0.01
    cache_dir: str = "out/integral-label-cache"
    label_timeout_seconds: float = 5
    label_max_attempts: int = 100
    log_interval: int = 20
    threads: int = 4
    device: str = "cpu"
    precision: str = "float32"

    def validate(self):
        if self.version != VERSION or self.condition not in CONDITIONS:
            raise ValueError("Unsupported data version or condition")
        for name in ("steps", "batch_size", "block_size", "n_layer", "n_head", "n_embd",
                     "eval_interval", "eval_examples", "final_examples", "baseline_examples", "eval_batch_size",
                     "generation_examples", "log_interval", "threads", "label_max_attempts"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.max_operators) is not int or not 0 <= self.max_operators <= 32:
            raise ValueError("max_operators must be an integer in [0,32]")
        if self.block_size < 2*self.max_operators+22:
            raise ValueError("Context must fit every complete record: block_size >= 2*max_operators+22")
        if self.n_embd % self.n_head:
            raise ValueError("Embedding width must be divisible by heads")
        if type(self.seed) is not int or type(self.eval_seed) is not int:
            raise ValueError("Seeds must be integers")
        for name in ("ramp_steps", "final_generation_examples", "generation_interval", "checkpoint_interval"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.ramp_steps and not 2 <= self.ramp_steps <= self.steps:
            raise ValueError("Explicit ramp_steps must be in [2, steps]")
        if self.generation_interval and self.generation_interval % self.eval_interval:
            raise ValueError("generation_interval must be a multiple of eval_interval")
        if self.generation_examples > self.eval_examples or (self.final_generation_examples or self.generation_examples) > self.final_examples:
            raise ValueError("Generation samples must fit their evaluation banks")
        if not 0 < self.ramp_fraction <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid ramp fraction or dropout")
        if type(self.warmup_steps) is not int or not 0 <= self.warmup_steps < self.steps:
            raise ValueError("Warmup must be nonnegative and shorter than training")
        for name in ("learning_rate", "weight_decay", "grad_clip", "label_timeout_seconds",
                     "absolute_tolerance", "relative_tolerance"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {name}")
        if min(self.learning_rate, self.grad_clip, self.label_timeout_seconds) <= 0:
            raise ValueError("Learning rate, gradient clip, and label timeout must be positive")
        if not self.eval_gammas or len(set(self.eval_gammas)) != len(self.eval_gammas):
            raise ValueError("Evaluation gammas must be nonempty and unique")
        if any(not math.isfinite(g) or g < 0 for g in (self.gamma_low, self.gamma_high, *self.eval_gammas)):
            raise ValueError("Gammas must be finite and nonnegative")
        if self.gamma_low > self.gamma_high:
            raise ValueError("gamma_low exceeds gamma_high")
        if type(self.pack_expressions) is not bool or not self.cache_dir:
            raise ValueError("Invalid packing or cache configuration")
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
    ramp = ramp_updates(config)
    indices = list(range(config.steps))
    if config.condition == "shuffled":
        random.Random(stream_seed(config.seed, "shuffle")).shuffle(indices)
    return [BatchSpec(i, config.gamma_high if config.condition == "hard" else
                      config.gamma_low+(config.gamma_high-config.gamma_low)*min(i/(ramp-1), 1),
                      stream_seed(config.seed, "optimizer-hard-batch" if config.condition == "hard" else
                                  "optimizer-batch", i)) for i in indices]


def learning_rate(config, step):
    return config.learning_rate*min((step+1)/max(config.warmup_steps, 1), 1)


def ramp_updates(config):
    return config.ramp_steps or max(2, math.ceil(config.steps*config.ramp_fraction))
