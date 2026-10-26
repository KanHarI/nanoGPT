"""Calibrate equal-time integer-ceiling stages for a matched two-condition run."""

import argparse
from dataclasses import replace
from pathlib import Path

from .benchmark import measure, staged_config
from .config import Config
from .corpus import Corpus, write_json


def calibrate(config, data_dir, out_dir, minutes=120, iterations=80, allow_cpu=False):
    config.validate()
    if config.training_gamma_sampling != "uniform_integer":
        raise ValueError("Stage calibration requires uniform per-sample gamma")
    if config.device != "cuda" and not (allow_cpu and config.device == "cpu"):
        raise ValueError("Use CUDA; --allow-cpu is only for pipeline smoke tests")
    corpus = Corpus(data_dir, config, require_training=False, verify_all=True)
    if corpus.manifest["kind"] != "pilot":
        raise ValueError("Use independent pilot data")
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    report = dict(base_config=config.to_dict(), minutes_per_condition=minutes,
                  conditions=["curriculum", "shuffled"], data_fingerprint=corpus.fingerprint,
                  stage_timings={})
    def checked(pool, count):
        result = measure(config, pool, warmup=10, iterations=count)
        if result["status"] != "ok" or result["memory_fraction"] > .9:
            raise RuntimeError("Calibration failed finite-loss/memory-headroom checks")
        return result
    report["stress"] = checked(corpus.named("pilot/stress"), min(iterations, 20))
    timings = []
    for gamma in range(int(config.gamma_low), int(config.gamma_high)+1):
        result = checked(corpus.named(f"pilot/{gamma}"), iterations)
        report["stage_timings"][str(gamma)] = result
        timings.append((gamma, result["training_seconds_per_update"]))
        write_json(root/"calibration.json", report)
        print(f"ceiling={gamma}: {result['training_seconds_per_update']:.6f} training seconds/update", flush=True)
    # Evaluation is deliberately larger than the timing pilot's banks.
    base = replace(config, eval_examples=128, final_examples=512, baseline_examples=512,
                   generation_examples=16, final_generation_examples=64, eval_batch_size=64)
    chosen = staged_config(base, timings, minutes)
    report.update(config=chosen.to_dict(), expected_training_seconds=sum(
        n*t for n,(_,t) in zip(chosen.gamma_stage_steps,timings)),
        training_examples_per_condition=chosen.steps*chosen.batch_size,
        corpus_training_streams=["curriculum"], exact_shuffled_replay=True)
    write_json(root/"calibration.json", report)
    write_json(root/"config.json", chosen.to_dict())
    print(f"Frozen {chosen.steps} updates, {len(chosen.gamma_stage_steps)} equal-time stages, "
          f"{chosen.steps*chosen.batch_size} fresh training examples; curriculum then shuffled", flush=True)
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--minutes", type=float, default=120)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args()
    calibrate(Config.load(args.config), args.data_dir, args.out_dir, args.minutes, args.iterations, args.allow_cpu)


if __name__ == "__main__":
    main()
