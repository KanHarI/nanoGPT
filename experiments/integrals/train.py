"""Resumable, full-sequence next-token training on numerical integrals."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import torch

from model import GPT, GPTConfig
from experiments.finite_field.runtime import autocast, configure
from .config import CONDITIONS, Config, batch_plan, learning_rate, ramp_updates
from .corpus import Corpus, Prefetch
from .data import Dataset, IGNORE_INDEX, VOCAB, batch
from .evaluate import Evaluation, loss_totals, summarize_losses
from .tracking import Tracker, add_arguments, from_arguments, grid_metrics


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def make_model(config):
    return GPT(GPTConfig(n_layer=config.n_layer, n_head=config.n_head, n_embd=config.n_embd,
                         block_size=config.block_size, vocab_size=len(VOCAB),
                         dropout=config.dropout, bias=False)).to(config.device)


def metadata(model):
    root = Path(__file__).resolve().parents[2]
    paths = [root/"model.py", root/"experiments/finite_field/runtime.py",
             root/"experiments/proposals/integral_ast_preview.py",
             root/"experiments/proposals/integral_distribution.py",
             *sorted(Path(__file__).parent.rglob("*.py"))]
    def git(*args):
        return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False).stdout.strip()
    return dict(python=platform.python_version(), torch=str(torch.__version__), cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                parameters=sum(p.numel() for p in model.parameters()), platform=platform.platform(),
                commit=git("rev-parse", "HEAD"), branch=git("branch", "--show-current"),
                source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})


def synchronize(config):
    if config.device == "cuda":
        torch.cuda.synchronize()
    elif config.device == "mps":
        torch.mps.synchronize()


def run(config, out_dir, resume=False, stop_after=None, tracking=None, data_dir=None, checkpoint_callback=None):
    config.validate()
    if config.device == "cuda" and data_dir is None:
        raise ValueError("CUDA training requires --data-dir with a complete prepared corpus")
    dataset = Corpus(data_dir, config) if data_dir else Dataset(
        config.cache_dir, config.max_operators, config.label_timeout_seconds, config.label_max_attempts)
    data_fingerprint = dataset.fingerprint if data_dir else None
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after <= config.steps):
        raise ValueError("stop_after must be in [1, steps]; original schedule is retained")
    directory = Path(out_dir)
    checkpoint_path, metrics_path = directory/"checkpoint.pt", directory/"metrics.jsonl"
    history_path = directory/"wandb_history.jsonl"
    if not resume and directory.exists() and any(directory.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {directory}; use --resume or a fresh directory")
    checkpoint = None
    if resume:
        # Only load trusted local checkpoints: optimizer and RNG state are included.
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if checkpoint["config"] != config.to_dict() or checkpoint["vocabulary"] != list(VOCAB):
            raise ValueError("Resume requires the original configuration and vocabulary")
        if checkpoint.get("data_fingerprint") != data_fingerprint:
            raise ValueError("Resume requires the same immutable data manifest")
        if stop_after is not None and stop_after < checkpoint["step"]:
            raise ValueError("stop_after precedes checkpoint")
        for path, key in ((metrics_path, "records"), (history_path, "history_records")):
            if len(path.read_text().splitlines()) < checkpoint[key]:
                raise ValueError(f"{path.name} is shorter than its checkpoint")
    directory.mkdir(parents=True, exist_ok=True)
    with Tracker(tracking, config, directory, resume=resume) as tracker:
        # W&B initialization may consume RNG; model initialization remains paired.
        configure(config)
        torch.manual_seed(config.seed)
        model = make_model(config)
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (0.9, 0.95), config.device)
        plan = batch_plan(config)
        step = records = supervised_tokens = 0
        timings = dict(training=0.0, evaluation=0.0, labeling=0.0, loading=0.0)
        completed = False
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            torch.set_rng_state(checkpoint["torch_rng"])
            if config.device == "cuda":
                torch.cuda.set_rng_state_all(checkpoint["device_rng"])
            elif config.device == "mps":
                torch.mps.set_rng_state(checkpoint["device_rng"])
            step, records, supervised_tokens, timings = (checkpoint[k] for k in
                                                        ("step", "records", "supervised_tokens", "timings"))
            for path, key in ((metrics_path, "records"), (history_path, "history_records")):
                lines = path.read_text().splitlines()[:checkpoint[key]]
                path.write_text("\n".join(lines)+("\n" if lines else ""))
        else:
            write_json(directory/"config.json", config.to_dict())
            write_json(directory/"environment.json", metadata(model))
            write_json(directory/"vocabulary.json", list(VOCAB))
            write_json(directory/"dataset.json", dict(manifest_sha256=data_fingerprint,
                                                       data_dir=str(data_dir) if data_dir else None))

        def make_evaluation(final=False):
            print(f"Preparing {'final' if final else 'monitoring'} banks: "
                  f"{len(config.eval_gammas)} gammas, two independent splits", flush=True)
            started = time.perf_counter()
            evaluation = Evaluation(config, dataset, final=final)
            timings["loading" if data_dir else "labeling"] += time.perf_counter()-started
            return evaluation

        evaluation = make_evaluation()

        def log(record):
            nonlocal records
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, allow_nan=False)+"\n")
            records += 1

        def save():
            device_rng = (torch.cuda.get_rng_state_all() if config.device == "cuda" else
                          torch.mps.get_rng_state() if config.device == "mps" else None)
            state = dict(config=config.to_dict(), vocabulary=list(VOCAB), model=model.state_dict(),
                         optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(), device_rng=device_rng,
                         step=step, records=records, history_records=len(history_path.read_text().splitlines()),
                         supervised_tokens=supervised_tokens, timings=timings,
                         data_fingerprint=data_fingerprint, completed=completed)
            temporary = checkpoint_path.with_suffix(".tmp")
            torch.save(state, temporary)
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            if checkpoint_path.exists():
                previous_tmp = directory/"checkpoint-previous.tmp"
                previous_tmp.unlink(missing_ok=True)
                os.link(checkpoint_path, previous_tmp)
                previous_tmp.replace(directory/"checkpoint-previous.pt")
            temporary.replace(checkpoint_path)
            # Relay only committed history: a Spot restart can rewind newer log lines.
            write_json(directory/"checkpoint_progress.json", dict(
                step=step, records=records, history_records=state["history_records"], completed=completed))
            if checkpoint_callback:
                checkpoint_callback(directory)

        def evaluate(bank, phase="eval"):
            synchronize(config)
            started = time.perf_counter()
            decode_now = (phase == "final" or step % (config.generation_interval or config.eval_interval) == 0
                          or step in (ramp_updates(config), config.steps, stop_after))
            grid = bank.evaluate(model, config, include_generation=decode_now)
            synchronize(config)
            timings["evaluation"] += time.perf_counter()-started
            log(dict(kind=phase, step=step, loss_grid=grid, timings=dict(timings)))
            if phase == "eval":
                tracker.log(grid_metrics(grid, step))
            target = grid["test"][f"{config.gamma_high:g}"]
            print(f"{phase}: {config.condition} step={step}/{config.steps} gamma={config.gamma_high:g} "
                  f"answer_nll={target['answer_nll']:.4f} full_nll={target['full_nll']:.4f} "
                  f"valid_answers={target.get('valid_answer_rate', 'not decoded')}", flush=True)
            return grid

        if checkpoint is None:
            evaluate(evaluation)
            save()
        limit = config.steps if stop_after is None else stop_after
        model.train()
        batches = (iter(Prefetch(dataset, plan[step:limit], config.batch_size)) if data_dir else
                   ((spec, dataset.training_examples(spec, config.batch_size)) for spec in plan[step:limit]))
        while step < limit:
            started = time.perf_counter()
            spec, examples = next(batches)
            x, y = batch(examples, config.block_size, config.device, config.pack_expressions)
            synchronize(config)
            timings["loading" if data_dir else "labeling"] += time.perf_counter()-started
            started = time.perf_counter()
            lr = learning_rate(config, step)
            for group in optimizer.param_groups:
                group["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            with autocast(config):
                logits, loss = model(x, y)  # All real targets; ONLY padding uses ignore_index.
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError(f"Nonfinite loss/gradient at update {step}")
            optimizer.step()
            synchronize(config)
            timings["training"] += time.perf_counter()-started
            supervised_tokens += (y != IGNORE_INDEX).sum().item()
            step += 1
            if step % config.log_interval == 0 or step == limit:
                metrics = summarize_losses(loss_totals(logits.detach(), y))
                log(dict(kind="train", step=step, batch_id=spec.index, batch_seed=spec.seed, gamma=spec.gamma,
                         gamma_sampling=spec.gamma_sampling, sample_gammas=[e.gamma for e in examples],
                         batch_sha256=hashlib.sha256("".join(e.text for e in examples).encode()).hexdigest(),
                         learning_rate=lr, grad_norm=norm.item(), timings=dict(timings), **metrics,
                         supervised_tokens=supervised_tokens, packed_rows=x.shape[0], padded_tokens=x.numel(),
                         mean_operators=sum(e.operators for e in examples)/len(examples)))
                tracker.log(dict(optimizer_step=step, **{f"train/{k}":v for k, v in metrics.items()},
                                 **{"train/gamma":spec.gamma, "train/learning_rate":lr,
                                    "train/grad_norm":norm.item(),
                                    **{f"time/{k}_seconds":v for k, v in timings.items()}}))
            if step % config.eval_interval == 0 or step in (ramp_updates(config), limit):
                evaluate(evaluation)
            if step % (config.checkpoint_interval or config.eval_interval) == 0 or step in (ramp_updates(config), limit):
                save()
        batches.close()
        final = None
        if step == config.steps:
            grid = evaluate(make_evaluation(final=True), phase="final")
            final = dict(step=step, config=config.to_dict(), parameters=sum(p.numel() for p in model.parameters()),
                         loss_grid=grid, timings=timings, supervised_tokens=supervised_tokens,
                         label_cache_stats=dict(dataset.stats), data_fingerprint=data_fingerprint, phase="final")
            write_json(directory/"final.json", final)
        tracker.finish_training(step, final)
        save()
    # Mark completion only after the SDK's finish/flush has returned successfully.
    write_json(directory/"tracking_complete.json", dict(step=step, sdk_finished=True))
    completed = final is not None
    save()
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
    parser.add_argument("--data-dir", help="Complete immutable corpus; mandatory for CUDA")
    add_arguments(parser)
    args = parser.parse_args()
    path = args.config or (Path(args.out_dir)/"config.json" if args.resume else None)
    config = Config.load(path) if path else Config()
    for name in ("condition", "seed", "device"):
        if getattr(args, name) is not None:
            setattr(config, name, getattr(args, name))
    run(config, args.out_dir, args.resume, args.stop_after, from_arguments(args), args.data_dir)


if __name__ == "__main__":
    main()
