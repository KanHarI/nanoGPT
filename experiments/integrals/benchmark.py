"""Benchmark cached pilot batches and freeze a 50-minute ramp/50-minute plateau."""

import argparse
from dataclasses import replace
import gc
import math
from pathlib import Path
import statistics
import time

import torch

from experiments.finite_field.runtime import autocast, configure
from .config import Config
from .corpus import Corpus, write_json
from .data import IGNORE_INDEX, batch
from .evaluate import Evaluation
from .train import make_model, metadata, synchronize


def choose_candidate(results, tolerance=0.05):
    usable = [row for row in results if row["status"] == "ok" and row["memory_fraction"] <= 0.9]
    if not usable:
        raise RuntimeError("No batch/context candidate passed all probes with 10% memory headroom")
    best = max(row["examples_per_second"] for row in usable)
    return min((r for r in usable if r["examples_per_second"] >= best*(1-tolerance)),
               key=lambda r:(r["batch_size"], -r["examples_per_second"]))


def calibrated_config(config, timings, ramp_minutes, plateau_minutes, loss_seconds, decode_seconds):
    if len(timings) < 2 or min(ramp_minutes, plateau_minutes) <= 0 or config.gamma_high <= config.gamma_low:
        raise ValueError("Need at least two gamma timings and positive phase durations")
    points = sorted(timings)
    if points[0][0] != config.gamma_low or points[-1][0] != config.gamma_high:
        raise ValueError("Calibration must cover both endpoints of the gamma ramp")
    if any(not math.isfinite(t) or t <= 0 for _,t in points):
        raise ValueError("Invalid measured training duration")
    mean = sum((b-a)*(ta+tb)/2 for (a,ta),(b,tb) in zip(points,points[1:]))/(config.gamma_high-config.gamma_low)
    ramp = max(2, math.ceil(60*ramp_minutes/mean))
    plateau = max(1, math.ceil(60*plateau_minutes/points[-1][1]))
    steps = ramp+plateau
    average = (ramp*mean+plateau*points[-1][1])/steps
    # Reserve at most ~5% active training time for each evaluation component.
    interval = max(1, math.ceil(max(300., 20*loss_seconds)/average))
    decode_interval = interval*max(1, math.ceil(max(1200., 20*decode_seconds)/(average*interval)))
    return replace(config, steps=steps, ramp_steps=ramp, ramp_fraction=ramp/steps,
                   warmup_steps=min(steps-1, max(1, round(steps*.01))), eval_interval=interval,
                   generation_interval=decode_interval, checkpoint_interval=max(1, round(300/average)),
                   log_interval=max(1, round(10/average))).validate()


def measure(config, examples, warmup=3, iterations=8):
    if len(examples) < config.batch_size or warmup < 1 or iterations < 1:
        raise ValueError("Pilot bank too small or invalid measurement counts")
    configure(config)
    torch.manual_seed(config.seed)
    model = optimizer = logits = loss = x = y = None
    try:
        model = make_model(config)
        optimizer = model.configure_optimizers(config.weight_decay, config.learning_rate, (.9,.95), config.device)
        if config.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        total, compute, packing, token_counts, rows = [], [], [], [], []
        for index in range(warmup+iterations):
            # Reuse only benchmark samples; these weights never enter training.
            offset = (index*config.batch_size) % len(examples)
            selected = (examples+examples)[offset:offset+config.batch_size]
            synchronize(config)
            started = time.perf_counter()
            x,y = batch(selected, config.block_size, config.device, config.pack_expressions)
            synchronize(config)
            packed = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            with autocast(config):
                logits,loss = model(x,y)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise FloatingPointError("Nonfinite pilot loss/gradient")
            optimizer.step()
            synchronize(config)
            finished = time.perf_counter()
            if index >= warmup:
                total.append(finished-started)
                compute.append(finished-packed)
                packing.append(packed-started)
                token_counts.append((y != IGNORE_INDEX).sum().item())
                rows.append(x.shape[0])
        reserved = torch.cuda.max_memory_reserved() if config.device == "cuda" else 0
        allocated = torch.cuda.max_memory_allocated() if config.device == "cuda" else 0
        capacity = torch.cuda.get_device_properties(0).total_memory if config.device == "cuda" else 1
        return dict(status="ok", seconds_per_update=statistics.median(total),
                    training_seconds_per_update=statistics.mean(compute), packing_seconds=statistics.mean(packing),
                    examples_per_second=config.batch_size/statistics.median(total),
                    useful_tokens_per_second=sum(token_counts)/sum(total),
                    packing_utilization=sum(token_counts)/(sum(rows)*config.block_size),
                    peak_reserved_bytes=reserved, peak_allocated_bytes=allocated,
                    memory_fraction=reserved/capacity, measured_updates=iterations)
    except torch.OutOfMemoryError:
        return dict(status="oom", memory_fraction=1.0)
    finally:
        model = optimizer = logits = loss = x = y = None
        gc.collect()
        if config.device == "cuda":
            torch.cuda.empty_cache()


def benchmark(config, data_dir, out_dir, batches=(64,128,256,512,1024), contexts=(128,256),
              ramp_minutes=50, plateau_minutes=50, iterations=8, allow_cpu=False):
    config.validate()
    if config.device != "cuda" and not (allow_cpu and config.device == "cpu"):
        raise ValueError("Use the target CUDA device; --allow-cpu is only a pipeline smoke")
    directory = Path(out_dir)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("Use a new benchmark output directory")
    directory.mkdir(parents=True, exist_ok=True)
    corpus = Corpus(data_dir, config, require_training=False, verify_all=True)
    if corpus.manifest["kind"] != "pilot":
        raise ValueError("Benchmark requires independent pilot data")
    gammas = sorted(set([*config.eval_gammas, config.gamma_low, config.gamma_high]))
    pools = {f"{g:g}":corpus.named(f"pilot/{g:g}") for g in gammas}
    pools["stress"] = corpus.named("pilot/stress")
    probes = list(dict.fromkeys([f"{gammas[0]:g}", f"{gammas[len(gammas)//2]:g}", f"{gammas[-1]:g}", "stress"]))
    report = dict(base_config=config.to_dict(), data_fingerprint=corpus.fingerprint,
                  gpu_calibration=config.device == "cuda", candidates=[])
    def save():
        write_json(directory/"benchmark.json", report)
    for context in sorted(set(contexts)):
        for size in sorted(set(batches)):
            candidate = replace(config, block_size=context, batch_size=size).validate()
            measurements = {}
            for probe in probes:
                measurements[probe] = measure(candidate, pools[probe], iterations=iterations)
                if measurements[probe]["status"] != "ok" or measurements[probe]["memory_fraction"] > .9:
                    break
            passed = len(measurements) == len(probes) and all(r["status"] == "ok" and r["memory_fraction"] <= .9
                                                                           for r in measurements.values())
            row = dict(batch_size=size, block_size=context, status="ok" if passed else "failed",
                       probes=measurements, memory_fraction=max(r["memory_fraction"] for r in measurements.values()))
            if passed:
                row["examples_per_second"] = size/statistics.mean(measurements[g]["seconds_per_update"] for g in probes if g != "stress")
            report["candidates"].append(row)
            save()
            print(f"context={context} batch={size}: {row['status']}", flush=True)
            if not passed:
                break
    selected = choose_candidate(report["candidates"])
    chosen = replace(config, block_size=selected["block_size"], batch_size=selected["batch_size"],
                     eval_batch_size=min(config.eval_batch_size, selected["batch_size"]))
    report["winner_repeat"] = measure(chosen, pools["stress"], iterations=iterations*2)
    if report["winner_repeat"]["status"] != "ok" or report["winner_repeat"]["memory_fraction"] > .9:
        raise RuntimeError("Winner failed longer stress verification")
    timings, report["gamma_timings"] = [], {}
    for gamma in gammas:
        measured = measure(chosen, pools[f"{gamma:g}"], iterations=iterations)
        if measured["status"] != "ok" or measured["memory_fraction"] > .9:
            raise RuntimeError(f"Selected batch failed at gamma {gamma}")
        report["gamma_timings"][str(gamma)] = measured
        timings.append((gamma, measured["training_seconds_per_update"]))
        save()
    evaluation = Evaluation(chosen, corpus)
    model = make_model(chosen)
    report["environment"] = metadata(model)
    synchronize(chosen)
    started = time.perf_counter()
    evaluation.evaluate(model, chosen, include_generation=False)
    synchronize(chosen)
    loss_seconds = time.perf_counter()-started
    started = time.perf_counter()
    evaluation.evaluate(model, chosen, include_generation=True)
    synchronize(chosen)
    full_seconds = time.perf_counter()-started
    decode_seconds = max(0, full_seconds-loss_seconds)
    chosen = calibrated_config(chosen, timings, ramp_minutes, plateau_minutes, loss_seconds, decode_seconds)
    report.update(chosen_config=chosen.to_dict(), loss_eval_seconds=loss_seconds,
                  additional_decode_seconds=decode_seconds, ramp_minutes=ramp_minutes, plateau_minutes=plateau_minutes)
    save()
    write_json(directory/"config.json", chosen.to_dict())
    print(f"Frozen {chosen.steps} updates, ramp={chosen.ramp_steps}, batch={chosen.batch_size}, context={chosen.block_size}")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="experiments/integrals/configs/gpu_64layer.json")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[64,128,256,512,1024])
    parser.add_argument("--contexts", type=int, nargs="+", default=[128,256])
    parser.add_argument("--ramp-minutes", type=float, default=50)
    parser.add_argument("--plateau-minutes", type=float, default=50)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    benchmark(Config.load(args.config), args.data_dir, args.out_dir, args.batches, args.contexts,
              args.ramp_minutes, args.plateau_minutes, args.iterations, args.allow_cpu)


if __name__ == "__main__":
    main()
