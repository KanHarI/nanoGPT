"""Validated experiment settings and the three matched gamma schedules."""

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random


@dataclass
class Config:
    condition: str = "curriculum"
    seed: int = 0
    include_power: bool = False
    include_division: bool = False
    gamma_low: float = 1.0
    gamma_high: float = 16.0
    ramp_fraction: float = 0.4
    max_operators: int = 32
    steps: int = 1000
    batch_size: int = 32
    block_size: int = 72
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    learning_rate: float = 0.0006
    lr_schedule: str = "constant"
    min_lr_ratio: float = 0.1
    warmup_steps: int = 20
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_per_field: int = 64
    final_per_field: int = 256
    baseline_per_field: int = 256
    eval_seed: int = 20261016
    eval_gammas: tuple = (0.0, 1.0, 4.0, 16.0)
    log_interval: int = 20
    threads: int = 4
    device: str = "cpu"
    precision: str = "float32"
    pack_expressions: bool = False
    eval_mode: str = "legacy"
    eval_batch_size: int = 0

    def validate(self):
        if self.condition not in ("hard", "curriculum", "mixture"):
            raise ValueError("condition must be hard, curriculum, or mixture")
        if not isinstance(self.include_power, bool):
            raise ValueError("include_power must be a JSON boolean")
        if not isinstance(self.include_division, bool):
            raise ValueError("include_division must be a JSON boolean")
        if not 0 <= self.max_operators <= 128:
            raise ValueError("max_operators must be in [0, 128]")
        if self.block_size < 2 * self.max_operators + 5:
            raise ValueError("block_size must be >= 2 * max_operators + 5")
        for name in ("steps", "batch_size", "n_layer", "n_head", "n_embd", "threads",
                     "eval_interval", "eval_per_field", "final_per_field",
                     "baseline_per_field", "log_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if not 0 < self.ramp_fraction <= 1:
            raise ValueError("ramp_fraction must be in (0, 1]")
        if not 0 <= self.dropout < 1 or not 0 <= self.min_lr_ratio <= 1:
            raise ValueError("Invalid dropout or minimum learning rate ratio")
        if not 0 <= self.warmup_steps < self.steps:
            raise ValueError("warmup_steps must be nonnegative and less than steps")
        if self.learning_rate <= 0 or self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("Invalid optimizer settings")
        if self.lr_schedule not in ("constant", "cosine"):
            raise ValueError("lr_schedule must be constant or cosine")
        for gamma in (self.gamma_low, self.gamma_high, *self.eval_gammas):
            if not math.isfinite(gamma) or gamma < 0:
                raise ValueError("All gammas must be finite and nonnegative")
        if self.gamma_low > self.gamma_high:
            raise ValueError("gamma_low cannot exceed gamma_high")
        if self.device not in ("cpu", "cuda", "mps"):
            raise ValueError("device must be cpu, cuda, or mps")
        if self.precision not in ("float32", "bfloat16"):
            raise ValueError("precision must be float32 or bfloat16")
        if self.precision == "bfloat16" and self.device == "mps":
            raise ValueError("bfloat16 training is supported on CPU/CUDA only")
        if not isinstance(self.pack_expressions, bool):
            raise ValueError("pack_expressions must be a boolean")
        if self.eval_mode not in ("legacy", "loss_grid"):
            raise ValueError("eval_mode must be legacy or loss_grid")
        if not isinstance(self.eval_batch_size, int) or self.eval_batch_size < 0:
            raise ValueError("eval_batch_size must be a nonnegative integer")
        return self

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text())).validate()

    @classmethod
    def from_saved(cls, values):
        """Saved runs before the division toggle always included division."""
        values = dict(values)
        values.setdefault("include_division", True)
        values.setdefault("lr_schedule", "cosine")
        return cls(**values).validate()

    @classmethod
    def load_saved(cls, path):
        return cls.from_saved(json.loads(Path(path).read_text()))


def gamma_schedule(config: Config) -> list[float]:
    if config.condition == "hard":
        return [config.gamma_high] * config.steps
    ramp_steps = max(2, math.ceil(config.steps * config.ramp_fraction))
    values = [config.gamma_low + (config.gamma_high - config.gamma_low)
              * min(step / (ramp_steps - 1), 1.0) for step in range(config.steps)]
    if config.condition == "mixture":
        # Exactly the curriculum's gamma multiset, shuffled by UPDATE. A batch
        # has one gamma in all arms, so within-batch heterogeneity is matched.
        random.Random(config.seed + 700_001).shuffle(values)
    return values


def learning_rate(config: Config, step: int) -> float:
    if step < config.warmup_steps:
        return config.learning_rate * (step + 1) / max(1, config.warmup_steps)
    if config.lr_schedule == "constant":
        return config.learning_rate
    progress = (step - config.warmup_steps) / max(1, config.steps - config.warmup_steps - 1)
    factor = config.min_lr_ratio + (1 - config.min_lr_ratio) * (1 + math.cos(math.pi * progress)) / 2
    return config.learning_rate * factor
