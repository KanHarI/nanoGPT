"""Shared precision and device setup for training, evaluation, and benchmarks."""

from contextlib import nullcontext
import torch


def autocast(config):
    if config.precision == "float32":
        return nullcontext()
    return torch.autocast(device_type=config.device, dtype=torch.bfloat16)


def configure(config):
    torch.set_num_threads(config.threads)
    if config.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this configuration")
        if config.precision == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This GPU does not support bfloat16")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
