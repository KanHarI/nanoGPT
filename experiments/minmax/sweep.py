"""Run matched conditions and compare only complete final reports."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics

from .config import CONDITIONS, Config
from .data import stream_seed
from .tracking import add_arguments, from_arguments
from .train import run, write_json


def compare(directories, output):
    runs, reference = {}, None
    for directory in map(Path, directories):
        report = json.loads((directory / "final.json").read_text())
        config = Config(**report["config"]).validate()
        if report["step"] != config.steps or report.get("phase") != "final":
            raise ValueError("Only complete fresh final evaluations can be compared")
        comparable = {k: v for k, v in config.to_dict().items() if k not in ("condition", "seed")}
        if reference is not None and comparable != reference:
            raise ValueError("Runs differ in settings other than seed and condition")
        reference = comparable
        key = config.condition, config.seed
        if key in runs:
            raise ValueError("Duplicate condition/seed")
        target = report["loss_grid"]["test"][f"{config.gamma_high:g}"]
        if (target["count"] != config.final_examples or
                target["seed"] != stream_seed(config.eval_seed, "final", f"{config.gamma_high:g}", "test")):
            raise ValueError("Final sample count/seed mismatch")
        runs[key] = target
    if not runs:
        raise ValueError("No runs supplied")
    conditions = sorted({condition for condition, _ in runs})
    seeds = sorted({seed for _, seed in runs})
    if any((condition, seed) not in runs for condition in conditions for seed in seeds):
        raise ValueError("All conditions must have the same seed set")
    summary = dict(config=reference, seeds=seeds, conditions={}, curriculum_minus_control={})
    for condition in conditions:
        summary["conditions"][condition] = dict(
            mean_loss=statistics.mean(runs[condition, seed]["loss"] for seed in seeds),
            mean_accuracy=statistics.mean(runs[condition, seed]["accuracy"] for seed in seeds),
            per_seed={str(seed): {name: runs[condition, seed][name] for name in ("loss", "accuracy", "baselines")}
                      for seed in seeds})
    if "curriculum" in conditions:
        for condition in conditions:
            if condition != "curriculum":
                differences = [runs["curriculum", seed]["loss"] - runs[condition, seed]["loss"] for seed in seeds]
                summary["curriculum_minus_control"][condition] = dict(mean=statistics.mean(differences),
                                                                      per_seed=differences, seeds=len(seeds))
    write_json(output, summary)
    return summary


def completed(directory, config):
    directory = Path(directory)
    if not (directory / "checkpoint.pt").exists() or not (directory / "final.json").exists():
        return False
    report = json.loads((directory / "final.json").read_text())
    return report.get("phase") == "final" and report["step"] == config.steps and report["config"] == config.to_dict()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--conditions", choices=CONDITIONS, nargs="+", default=["curriculum", "hard", "shuffled"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--resume", action="store_true")
    add_arguments(parser)
    args = parser.parse_args()
    if len(set(args.conditions)) != len(args.conditions) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Conditions and seeds must be unique")
    base, directories = Config.load(args.config), []
    for seed in args.seeds:
        for condition in args.conditions:
            config = replace(base, seed=seed, condition=condition)
            directory = Path(args.out_dir) / f"{condition}-seed{seed}"
            if not (args.resume and completed(directory, config)):
                run(config, directory, resume=args.resume and (directory / "checkpoint.pt").exists(),
                    tracking=from_arguments(args))
            directories.append(directory)
    summary = compare(directories, Path(args.out_dir) / "summary.json")
    for condition, result in summary["conditions"].items():
        print(f"{condition}: target loss={result['mean_loss']:.4f}, accuracy={result['mean_accuracy']:.3f}")


if __name__ == "__main__":
    main()
