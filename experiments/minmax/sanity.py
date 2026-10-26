"""Overfit a fixed sample to validate the complete answer-only learning path."""

import argparse
import json

import torch

from experiments.finite_field.runtime import configure
from .config import Config
from .data import Generator, IGNORE_INDEX, batch
from .train import make_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    config = Config(n_layer=2, n_head=4, n_embd=64, threads=2)
    configure(config)
    torch.manual_seed(17)
    sampler = Generator(29, max_operators=4)
    examples = [sampler.sample(2) for _ in range(16)]
    x, y = batch(examples, config.block_size)
    model = make_model(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = model(x, y)[1].item()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits, loss = model(x, y)
        selected = y != IGNORE_INDEX
        accuracy = (logits[selected].argmax(-1) == y[selected]).float().mean().item()
    print(json.dumps(dict(initial_loss=initial, final_loss=loss.item(), accuracy=accuracy,
                          examples=len(examples), steps=args.steps)))
    if accuracy < 0.95 or loss.item() > 0.1:
        raise RuntimeError("Tiny-sample overfitting check failed")


if __name__ == "__main__":
    main()
