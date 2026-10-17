"""Overfit a tiny, fixed sample to check the answer-only training pipeline."""

import json
import torch
from model import GPT, GPTConfig
from .data import Generator, IGNORE_INDEX, VOCAB, batch


def main():
    torch.set_num_threads(2)
    torch.manual_seed(17)
    sampler = Generator(29, max_operators=3)
    examples = [sampler.sample(1.0, prime=7) for _ in range(16)]
    x, y = batch(examples, 16)
    model = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=64, block_size=16,
                          vocab_size=len(VOCAB), dropout=0.0, bias=False))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    initial = model(x, y)[1].item()
    for _ in range(250):
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
                          examples=len(examples), steps=250)))
    if accuracy < 0.95 or loss.item() > 0.1:
        raise RuntimeError("Tiny-sample overfitting check failed")


if __name__ == "__main__":
    main()
