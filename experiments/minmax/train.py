"""Resumable answer-only training for the signed min/max task."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch

from model import GPT, GPTConfig
from experiments.finite_field.runtime import autocast, configure
from .config import CONDITIONS, Config, batch_plan, generate_batch, learning_rate
from .data import IGNORE_INDEX, VOCAB, batch
from .evaluate import Evaluation, structural_checks
from .tracking import Tracker, add_arguments, from_arguments


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def make_model(config):
    return GPT(GPTConfig(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
                         block_size=config.block_size, vocab_size=len(VOCAB),
                         dropout=config.dropout, bias=False)).to(config.device)


def metadata(model):
    root = Path(__file__).resolve().parents[2]
    paths = [root / "model.py", root / "experiments/finite_field/data.py",
             root / "experiments/finite_field/runtime.py", *sorted(Path(__file__).parent.rglob("*.py"))]
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False).stdout.strip()
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                platform=platform.platform(), parameters=sum(p.numel() for p in model.parameters()),
                commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                working_tree=git("status", "--short"),
                source_sha256={str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def run(config, out_dir, resume=False, stop_after=None, tracking=None):
    config.validate()
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= config.steps):
        raise ValueError("stop_after must be in [1, steps]; the original schedule is retained")
    directory = Path(out_dir)
    checkpoint_path = directory / "checkpoint.pt"
    metrics_path = directory / "metrics.jsonl"
    if not resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {directory}; use --resume or another directory")
    checkpoint = None
    if resume:
        # Load only trusted local checkpoints: Python RNG/optimizer state is included.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config.to_dict():
            raise ValueError("Resume requires the original full configuration")
        if stop_after is not None and stop_after < checkpoint["step"]:
            raise ValueError("stop_after precedes the checkpoint")
        if len(metrics_path.read_text().splitlines()) < checkpoint["records"]:
            raise ValueError("Metrics log is shorter than its checkpoint")
    directory.mkdir(parents=True, exist_ok=True)
    # SDK initialization may consume RNG. Seed/restore only after entering Tracker.
    with Tracker(tracking, config, directory, resume=resume) as tracker:
        configure(config)
        torch.manual_seed(config.seed)
        model = make_model(config)
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (0.9, 0.95), config.device)
        plan = batch_plan(config)
        step, records, train_seconds, characters = 0, 0, 0.0, 0
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if config.device == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["device_rng"])
            elif config.device == "mps":
                torch.mps.set_rng_state(checkpoint["device_rng"])
            step, records, train_seconds, characters = (checkpoint[k] for k in
                                                        ("step", "records", "train_seconds", "characters"))
            lines = metrics_path.read_text().splitlines()[:records]
            metrics_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        else:
            write_json(directory / "config.json", config.to_dict())
            write_json(directory / "environment.json", metadata(model))
        evaluation = Evaluation(config)

        def log(record):
            nonlocal records
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False) + "\n")
            records += 1

        def save():
            device_rng = (torch.cuda.get_rng_state_all() if config.device == "cuda" else
                          torch.mps.get_rng_state() if config.device == "mps" else None)
            state = dict(config=config.to_dict(), model=model.state_dict(), optimizer=optimizer.state_dict(),
                         torch_rng=torch.get_rng_state(), device_rng=device_rng, step=step, records=records,
                         train_seconds=train_seconds, characters=characters)
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint_path)

        def evaluate():
            started = time.perf_counter()
            grid = evaluation.evaluate(model, config)
            log(dict(kind="eval", step=step, train_seconds=train_seconds,
                     eval_seconds=time.perf_counter() - started, loss_grid=grid))
            tracker.log(grid, step)
            target = grid["test"][f"{config.gamma_high:g}"]
            print(f"{config.condition} seed={config.seed} step={step}/{config.steps} "
                  f"gamma={config.gamma_high:g} loss={target['loss']:.4f} accuracy={target['accuracy']:.3f}", flush=True)

        if checkpoint is None:
            evaluate()
            save()
        limit = config.steps if stop_after is None else stop_after
        model.train()
        while step < limit:
            started = time.perf_counter()
            spec = plan[step]
            examples = generate_batch(config, spec)
            x, y = batch(examples, config.block_size, config.device, config.pack_expressions)
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(config):
                logits, loss = model(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite loss/gradient at update {step}")
            optimizer.step()
            loss_value = loss.item()
            selected = y != IGNORE_INDEX
            accuracy = (logits.detach()[selected].argmax(-1) == y[selected]).float().mean().item()
            train_seconds += time.perf_counter() - started
            characters += sum(len(e.text) for e in examples)
            step += 1
            if step % config.log_interval == 0 or step == limit:
                log(dict(kind="train", step=step, batch_id=spec.index, batch_seed=spec.seed, gamma=spec.gamma,
                         loss=loss_value, accuracy=accuracy, learning_rate=lr, grad_norm=norm.item(),
                         train_seconds=train_seconds, supervised_answers=step * config.batch_size,
                         characters=characters, packed_rows=x.shape[0], input_tokens=x.numel(),
                         mean_operators=sum(e.operators for e in examples) / len(examples),
                         mean_depth=sum(e.depth for e in examples) / len(examples),
                         mean_stack_depth=sum(e.stack_depth for e in examples) / len(examples),
                         mean_negations=sum(e.negations for e in examples) / len(examples),
                         roots=dict(Counter(e.root for e in examples))))
            if step % config.eval_interval == 0 or step == limit:
                evaluate()
                save()
        final = None
        if step == config.steps:
            final = dict(step=step, config=config.to_dict(), train_seconds=train_seconds,
                         parameters=sum(p.numel() for p in model.parameters()), phase="final",
                         loss_grid=Evaluation(config, final=True).evaluate(model, config),
                         structural_checks=structural_checks(model, config))
            write_json(directory / "final.json", final)
        tracker.finish_training(step, final)
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--condition", choices=CONDITIONS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    add_arguments(parser)
    args = parser.parse_args()
    path = args.config or (Path(args.out_dir) / "config.json" if args.resume else None)
    config = Config.load(path) if path else Config()
    for name in ("condition", "seed", "device"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    run(config, args.out_dir, args.resume, args.stop_after, from_arguments(args))


if __name__ == "__main__":
    main()
