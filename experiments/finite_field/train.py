"""CPU-first, resumable training; all reported losses supervise answers only."""

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import torch

from model import GPT, GPTConfig
from .config import Config, gamma_schedule, learning_rate
from .data import Generator, IGNORE_INDEX, VOCAB, batch
from .evaluate import EvaluationSet, LossGrid
from .runtime import autocast, configure
from .tracking import Tracker, add_arguments as add_tracking_arguments, from_arguments as tracking_arguments


def metadata():
    def git(*args):
        result = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return result.stdout.strip()
    root = Path(__file__).resolve().parents[2]
    sources = [root / "model.py", *sorted(Path(__file__).parent.rglob("*.py"))]
    hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
              for path in sources}
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                platform=platform.platform(), processor=platform.processor(),
                commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                working_tree=git("status", "--short"), source_sha256=hashes,
                source_snapshot=json.loads((root / "source_snapshot.json").read_text())
                if (root / "source_snapshot.json").exists() else None)


def run(config: Config, out_dir, resume=False, stop_after=None, tracking=None, final_evaluation=False):
    config.validate()
    if stop_after is not None and not 1 <= stop_after <= config.steps:
        raise ValueError("stop_after must be in [1, steps]; it does not change the schedule")
    out_dir = Path(out_dir)
    checkpoint_path = out_dir / "checkpoint.pt"
    if not resume and out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out_dir}; use --resume or a new directory")
    if resume and not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = None
    if resume:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        # Validate locally before creating or resuming an external tracking run.
        saved_config = Config.from_saved(checkpoint["config"])
        if json.dumps(saved_config.to_dict(), sort_keys=True) != json.dumps(config.to_dict(), sort_keys=True):
            raise ValueError("Resume requires the original config, including total steps and seed")
        if stop_after is not None and stop_after < checkpoint["step"]:
            raise ValueError("stop_after precedes the saved checkpoint")
        if len((out_dir / "metrics.jsonl").read_text().splitlines()) < checkpoint["records"]:
            raise ValueError("Metrics log is shorter than its checkpoint")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Initialize tracking before seeding the model/restoring RNG state. Tracking
    # configuration is intentionally separate from the scientific configuration.
    with Tracker(tracking, config, out_dir, resume=resume) as tracker:
        return _train(config, out_dir, checkpoint, stop_after, tracker, final_evaluation)


def _train(config, out_dir, checkpoint, stop_after, tracker, final_evaluation):
    resume = checkpoint is not None
    checkpoint_path, metrics_path = out_dir / "checkpoint.pt", out_dir / "metrics.jsonl"
    configure(config)
    torch.manual_seed(config.seed)
    model_args = dict(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
                      block_size=config.block_size, vocab_size=len(VOCAB),
                      dropout=config.dropout, bias=False)
    model = GPT(GPTConfig(**model_args)).to(config.device)
    optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate,
                                           (0.9, 0.95), config.device)
    sampler = Generator(config.seed + 100_003, config.max_operators, config.include_power,
                        include_division=config.include_division)
    schedule = gamma_schedule(config)
    step, train_seconds, records = 0, 0.0, 0
    if resume:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        sampler.load_state_dict(checkpoint["sampler"])
        torch.set_rng_state(checkpoint["torch_rng"])
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["device_rng"])
        elif config.device == "mps":
            torch.mps.set_rng_state(checkpoint["device_rng"])
        step, train_seconds, records = (checkpoint[k] for k in ("step", "train_seconds", "records"))
        # Discard log rows beyond the last atomic checkpoint after a crash.
        lines = metrics_path.read_text().splitlines()
        metrics_path.write_text("\n".join(lines[:records]) + ("\n" if records else ""))
    else:
        (out_dir / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
        (out_dir / "environment.json").write_text(json.dumps(metadata(), indent=2) + "\n")

    evaluation = (LossGrid(config) if config.eval_mode == "loss_grid" else
                  EvaluationSet(config, config.gamma_high, config.eval_per_field, config.eval_seed))

    def log(record):
        nonlocal records
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(record) + "\n")
        records += 1
        tracker.log(record)

    def save():
        device_rng = None
        if config.device == "cuda":
            device_rng = torch.cuda.get_rng_state_all()
        elif config.device == "mps":
            device_rng = torch.mps.get_rng_state()
        state = dict(config=config.to_dict(), model_args=model_args, model=model.state_dict(),
                     optimizer=optimizer.state_dict(), sampler=sampler.state_dict(),
                     torch_rng=torch.get_rng_state(), device_rng=device_rng,
                     step=step, train_seconds=train_seconds, records=records)
        temporary = out_dir / "checkpoint.tmp"
        torch.save(state, temporary)
        temporary.replace(checkpoint_path)

    def evaluate():
        result = evaluation.evaluate(model, config)
        if config.eval_mode == "loss_grid":
            log(dict(kind="loss_grid", step=step, train_seconds=train_seconds,
                     supervised_answers=sampler.items, loss_grid=result))
            target = result["test"][f"{config.gamma_high:g}"]
            print(f"{config.condition} seed={config.seed} step={step}/{config.steps} "
                  f"gamma={config.gamma_high:g} test loss={target['loss']:.4f}", flush=True)
            return
        log(dict(kind="eval", step=step, train_seconds=train_seconds,
                 supervised_answers=sampler.items, **result))
        print(f"{config.condition} seed={config.seed} step={step}/{config.steps} "
              f"target loss={result['loss']:.4f} accuracy={result['accuracy']:.3f}", flush=True)

    if not resume:
        evaluate()
        save()
    limit = config.steps if stop_after is None else stop_after
    model.train()
    while step < limit:
        started = time.perf_counter()
        gamma = schedule[step]
        examples = [sampler.sample(gamma) for _ in range(config.batch_size)]
        x, y = batch(examples, config.block_size, config.device, pack=config.pack_expressions)
        lr = learning_rate(config, step)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        with autocast(config):
            logits, loss = model(x, y)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
        if not torch.isfinite(loss) or not torch.isfinite(grad_norm):
            raise FloatingPointError(f"Nonfinite training loss/gradient at step {step}")
        optimizer.step()
        # Scalar reads synchronize devices before timing; CPU is the default.
        loss_value = loss.item()
        selected = y != IGNORE_INDEX
        accuracy = (logits.detach()[selected].argmax(-1) == y[selected]).float().mean().item()
        train_seconds += time.perf_counter() - started
        step += 1
        if step % config.log_interval == 0 or step == limit:
            log(dict(kind="train", step=step, gamma=gamma, loss=loss_value, accuracy=accuracy,
                     learning_rate=lr, grad_norm=grad_norm.item(), train_seconds=train_seconds,
                     supervised_answers=sampler.items, characters=sampler.characters,
                     packed_rows=x.shape[0], input_tokens=x.numel(),
                     generation_attempts=sampler.attempts,
                     rejection_rate=1 - sampler.items / sampler.attempts,
                     mean_operators=sum(e.operators for e in examples) / len(examples),
                     mean_depth=sum(e.depth for e in examples) / len(examples),
                     operator_counts=dict(sampler.operator_counts),
                     field_counts=dict(sampler.field_counts)))
        if step % config.eval_interval == 0 or step == limit:
            evaluate()
            save()
    if final_evaluation and step == config.steps:
        from .evaluate import evaluate_checkpoint
        report = evaluate_checkpoint(checkpoint_path, out_dir / "final.json")
        tracker.final(report)
    tracker.finish_training(step, checkpoint_path)
    return checkpoint_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON overrides of Config defaults")
    parser.add_argument("--condition", choices=("hard", "curriculum", "mixture"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--include-power", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-division", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--final-eval", action="store_true",
                        help="Evaluate the independent final sample when training completes")
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if args.resume and args.config is None:
        config = Config.load_saved(Path(args.out_dir) / "config.json")
    else:
        config = Config.load(args.config) if args.config else Config()
    for name in ("condition", "seed", "include_power", "include_division", "device"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    run(config, args.out_dir, args.resume, args.stop_after,
        tracking=tracking_arguments(args), final_evaluation=args.final_eval)


if __name__ == "__main__":
    main()
