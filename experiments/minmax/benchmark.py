"""Measure batch/context throughput on the target GPU and freeze a run recipe.

Batch size always counts supervised expressions, even when they share contexts.
Timings include fresh procedural generation, packing, forward/backward, and the
optimizer. No benchmark weights or optimizer state enter the actual experiment.
"""

import argparse
from dataclasses import replace
import gc
import json
import math
from pathlib import Path
import statistics
import time

import torch

from .config import Config
from .data import Generator, IGNORE_INDEX, batch
from experiments.finite_field.runtime import autocast, configure
from .evaluate import Evaluation
from .train import metadata, make_model


def choose_candidate(results, tolerance=0.05):
    usable = [row for row in results if row.get("status") == "ok" and row["memory_fraction"] <= 0.9]
    if not usable:
        raise RuntimeError("No benchmark candidate fits with 10% GPU memory headroom")
    best = max(row["answers_per_second"] for row in usable)
    # Prefer more updates over a huge batch when throughput is effectively tied.
    near_best = [row for row in usable if row["answers_per_second"] >= best * (1 - tolerance)]
    return min(near_best, key=lambda row: (row["batch_size"], -row["answers_per_second"]))


def measure(config, gamma, warmup=3, iterations=8, minimum_seconds=2.0):
    configure(config)
    torch.manual_seed(config.seed)
    model = optimizer = x = y = logits = loss = None
    try:
        model = make_model(config)
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate,
                                               (0.9, 0.95), config.device)
        generator = Generator(7009, config.max_operators, config.operator_weights)
        durations, generation_times, rows = [], [], []
        torch.cuda.reset_peak_memory_stats()
        index = 0
        while index < warmup + iterations or sum(durations) < minimum_seconds:
            torch.cuda.synchronize()
            started = time.perf_counter()
            examples = [generator.sample(gamma) for _ in range(config.batch_size)]
            x, y = batch(examples, config.block_size, config.device, pack=config.pack_expressions)
            generation_seconds = time.perf_counter() - started
            optimizer.zero_grad(set_to_none=True)
            with autocast(config):
                logits, loss = model(x, y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite benchmark loss/gradient")
            optimizer.step()
            # Include the same answer statistics and synchronizations as training.
            loss.item()
            selected = y != IGNORE_INDEX
            (logits.detach()[selected].argmax(-1) == y[selected]).float().mean().item()
            torch.cuda.synchronize()
            if index >= warmup:
                durations.append(time.perf_counter() - started)
                generation_times.append(generation_seconds)
                rows.append(x.shape[0])
            index += 1
        seconds = statistics.median(durations)
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        return dict(status="ok", batch_size=config.batch_size, block_size=config.block_size,
                    pack_expressions=config.pack_expressions, gamma=gamma,
                    seconds_per_update=seconds, mean_seconds_per_update=statistics.mean(durations),
                    answers_per_second=config.batch_size / seconds,
                    median_generation_seconds=statistics.median(generation_times),
                    mean_packed_rows=statistics.mean(rows), measured_updates=len(durations),
                    peak_allocated_bytes=allocated, peak_reserved_bytes=reserved,
                    memory_fraction=reserved / torch.cuda.get_device_properties(0).total_memory)
    except torch.OutOfMemoryError:
        return dict(status="oom", batch_size=config.batch_size, block_size=config.block_size,
                    pack_expressions=config.pack_expressions, gamma=gamma)
    finally:
        model = optimizer = x = y = logits = loss = None
        gc.collect()
        torch.cuda.empty_cache()


def benchmark(config, out_dir, batches, contexts, ramp_minutes=50, iterations=8):
    if config.device != "cuda":
        raise ValueError("This benchmark requires the actual CUDA training device")
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    configure(config)
    report = dict(gpu=torch.cuda.get_device_name(0), gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory,
                  base_config=config.to_dict(), candidates=[])

    model = make_model(config)
    report["environment"] = metadata(model)
    del model
    torch.cuda.empty_cache()

    def save():
        (directory / "benchmark.json").write_text(json.dumps(report, indent=2) + "\n")

    # Test hardest gamma first: it packs the fewest expressions and needs the
    # most memory. The local optimum is selected from measured candidates only.
    for context in contexts:
        for size in batches:
            candidate = replace(config, batch_size=size, block_size=context).validate()
            result = measure(candidate, config.gamma_high, iterations=iterations)
            report["candidates"].append(result)
            save()
            print(json.dumps(result), flush=True)
            if result["status"] == "oom" or result.get("memory_fraction", 1) > 0.9:
                break
    selected = choose_candidate(report["candidates"])
    chosen = replace(config, batch_size=selected["batch_size"], block_size=selected["block_size"])
    # Check packing against one-expression rows with the same answer budget.
    unpacked = measure(replace(chosen, pack_expressions=False), config.gamma_high, iterations=iterations)
    report["unpacked_reference"] = unpacked
    # Recheck the winner; do not use a single noisy timing as the schedule basis.
    repeated = measure(chosen, config.gamma_high, iterations=iterations * 2, minimum_seconds=5)
    report["winner_repeat"] = repeated
    if repeated["status"] != "ok" or repeated["memory_fraction"] > 0.9:
        raise RuntimeError("Winning candidate failed its longer verification")
    report["ramp_calibration"] = []
    for gamma in range(math.ceil(config.gamma_low), math.floor(config.gamma_high) + 1):
        result = measure(chosen, gamma, iterations=iterations)
        if result["status"] != "ok":
            raise RuntimeError(f"Selected batch failed at gamma {gamma}")
        report["ramp_calibration"].append(result)
        save()
    # Trapezoidal average approximates a continuous linear gamma ramp, including
    # its changing packing efficiency. This targets training time, excluding eval.
    times = [row["mean_seconds_per_update"] for row in report["ramp_calibration"]]
    average = (sum(times) - (times[0] + times[-1]) / 2) / (len(times) - 1) if len(times) > 1 else times[0]
    ramp_updates = max(2, math.ceil(ramp_minutes * 60 / average))
    steps = math.ceil(ramp_updates / config.ramp_fraction)
    high_seconds = report["ramp_calibration"][-1]["mean_seconds_per_update"]
    chosen = replace(chosen, steps=steps, warmup_steps=min(1000, max(1, steps // 100)),
                     eval_interval=max(1, round(120 / high_seconds)),
                     log_interval=max(1, round(10 / high_seconds)), eval_batch_size=chosen.batch_size)
    chosen.validate()
    # Measure the actual 24-bank monitoring pass separately from training.
    model = make_model(chosen)
    started = time.perf_counter()
    evaluation = Evaluation(chosen)
    bank_seconds = time.perf_counter() - started
    torch.cuda.synchronize()
    started = time.perf_counter()
    evaluation.evaluate(model, chosen)
    torch.cuda.synchronize()
    eval_seconds = time.perf_counter() - started
    del evaluation, model
    torch.cuda.empty_cache()
    report.update(selected=selected, ramp_minutes=ramp_minutes, ramp_updates=ramp_updates,
                  expected_ramp_training_seconds=ramp_updates * average,
                  expected_total_training_seconds=ramp_updates * average + (steps - ramp_updates) * high_seconds,
                  evaluation_bank_setup_seconds=bank_seconds, evaluation_seconds=eval_seconds,
                  expected_plateau_evaluation_fraction=eval_seconds / (eval_seconds + chosen.eval_interval * high_seconds),
                  chosen_config=chosen.to_dict())
    save()
    (directory / "config.json").write_text(json.dumps(chosen.to_dict(), indent=2) + "\n")
    print(f"Selected batch={chosen.batch_size}, context={chosen.block_size}, steps={steps}, ramp updates={ramp_updates}", flush=True)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/minmax/configs/gpu_24layer.json")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256, 512, 1024, 2048])
    parser.add_argument("--contexts", type=int, nargs="+", default=[128, 256])
    parser.add_argument("--ramp-minutes", type=float, default=50)
    args = parser.parse_args()
    benchmark(Config.load(args.config), args.out_dir, sorted(set(args.batches)), sorted(set(args.contexts)), args.ramp_minutes)


if __name__ == "__main__":
    main()
