"""Compare completed, matched runs and export figures plus paired statistics."""

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path

import numpy as np

from .config import Config


def paired_difference(values):
    """Seed-level bootstrap, never treating evaluation items as training repeats."""
    values = np.asarray(values, dtype=float)
    interval = None
    if len(values) >= 2:
        rng = np.random.default_rng(1949)
        means = rng.choice(values, size=(10_000, len(values)), replace=True).mean(axis=1)
        interval = np.quantile(means, [0.025, 0.975]).tolist()
    return dict(mean=float(values.mean()), per_seed=values.tolist(), ci95_bootstrap=interval,
                seeds=len(values))


def compare(run_dirs, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    runs, reference = {}, None
    for directory in map(Path, run_dirs):
        final = json.loads((directory / "final.json").read_text())
        config = Config.from_saved(final["config"]).to_dict()
        if final["step"] != config["steps"]:
            raise ValueError(f"Incomplete run: {directory}")
        comparable = {k: v for k, v in config.items() if k not in ("seed", "condition")}
        if reference is not None and comparable != reference:
            raise ValueError("Runs differ in settings other than seed/condition")
        reference = comparable
        key = config["condition"], config["seed"]
        if key in runs:
            raise ValueError(f"Duplicate condition/seed: {key}")
        target = next(r for r in final["evaluations"] if r["gamma"] == config["gamma_high"])
        if target["count"] != 8 * config["final_per_field"] or target["seed"] != config["eval_seed"] + 10_000_019:
            raise ValueError("Final evaluations must use the configured common sample size/seed")
        history = [json.loads(line) for line in (directory / "metrics.jsonl").read_text().splitlines()]
        evaluations = [r for r in history if r["kind"] == "eval"]
        if config["eval_mode"] == "loss_grid":
            evaluations = [dict(step=r["step"], train_seconds=r["train_seconds"],
                                **r["loss_grid"]["test"][f"{config['gamma_high']:g}"])
                           for r in history if r["kind"] == "loss_grid"]
        runs[key] = dict(target=target, final=final, history=evaluations)
    if not runs:
        raise ValueError("No runs provided")
    conditions = sorted({condition for condition, _ in runs})
    seeds = sorted({seed for _, seed in runs})
    if any((condition, seed) not in runs for condition in conditions for seed in seeds):
        raise ValueError("Every condition must have the same seed set")
    summary = dict(config=reference, seeds=seeds, conditions={}, paired_loss_differences={})
    for condition in conditions:
        summary["conditions"][condition] = {
            "mean_loss": float(np.mean([runs[condition, seed]["target"]["loss"] for seed in seeds])),
            "per_seed": {str(seed): {key: runs[condition, seed]["target"][key]
                                     for key in ("loss", "accuracy", "frequency_baseline_loss",
                                                 "frequency_baseline_accuracy") if key in runs[condition, seed]["target"]}
                         for seed in seeds}}
        if "accuracy" in runs[condition, seeds[0]]["target"]:
            summary["conditions"][condition]["mean_accuracy"] = float(np.mean(
                [runs[condition, seed]["target"]["accuracy"] for seed in seeds]))
    for first, second in (("curriculum", "hard"), ("curriculum", "mixture"), ("mixture", "hard")):
        if first in conditions and second in conditions:
            differences = [runs[first, seed]["target"]["loss"] - runs[second, seed]["target"]["loss"]
                           for seed in seeds]
            summary["paired_loss_differences"][f"{first}_minus_{second}"] = paired_difference(differences)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    os.environ.setdefault("MPLCONFIGDIR", str(out_dir.resolve() / "mpl-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"hard": "#4055b5", "curriculum": "#c54c36", "mixture": "#27845b"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for condition in conditions:
        by_step = defaultdict(list)
        for seed in seeds:
            history = runs[condition, seed]["history"]
            for row in history:
                by_step[row["step"]].append(row)
            axes[1].plot([r["train_seconds"] for r in history], [r["loss"] for r in history],
                         color=colors[condition], alpha=0.6, label=condition if seed == seeds[0] else None)
        # Only average checkpoints shared by all seeds (resumes may add others).
        steps = sorted(step for step, rows in by_step.items() if len(rows) == len(seeds))
        axes[0].plot([step * reference["batch_size"] for step in steps],
                     [np.mean([r["loss"] for r in by_step[step]]) for step in steps],
                     label=condition, color=colors[condition])
    for axis in axes:
        axis.set_ylabel(f"Answer cross-entropy at gamma={reference['gamma_high']:g}")
        axis.grid(alpha=0.2)
        axis.legend()
    axes[0].set_xlabel("Supervised training answers (mean over seeds)")
    axes[1].set_xlabel("Training seconds, including generation (each seed)")
    figure.savefig(out_dir / "learning_curves.png", dpi=160)
    figure.savefig(out_dir / "learning_curves.pdf")
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    for condition in conditions:
        by_gamma = defaultdict(list)
        for seed in seeds:
            for result in runs[condition, seed]["final"]["evaluations"]:
                by_gamma[result["gamma"]].append(result["loss"])
        gammas = sorted(by_gamma)
        axis.plot(gammas, [np.mean(by_gamma[g]) for g in gammas], marker="o",
                  color=colors[condition], label=condition)
    axis.set(xlabel="Evaluation gamma", ylabel="Final answer cross-entropy")
    axis.grid(alpha=0.2)
    axis.legend()
    figure.savefig(out_dir / "difficulty_sweep.png", dpi=160)
    plt.close(figure)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.run_dirs, args.out_dir), indent=2))


if __name__ == "__main__":
    main()
