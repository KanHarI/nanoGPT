"""Run matched conditions sequentially on the local machine, then compare."""

import argparse
from dataclasses import replace
import json
from pathlib import Path

from .compare import compare
from .config import Config
from .run_plan import active_conditions
from .train import run
from .tracking import add_arguments as add_tracking_arguments, from_arguments as tracking_arguments


def completed_run(directory, config):
    """Skip only a saved full-budget result matching the requested recipe."""
    directory = Path(directory)
    if not (directory / "checkpoint.pt").exists() or not (directory / "final.json").exists():
        return False
    try:
        report = json.loads((directory / "final.json").read_text())
        saved = Config.from_saved(report["config"])
        return (report["step"] == config.steps and
                json.dumps(saved.to_dict(), sort_keys=True) == json.dumps(config.to_dict(), sort_keys=True))
    except (ValueError, KeyError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--conditions", nargs="+", choices=("hard", "curriculum", "mixture"),
                        default=["hard", "curriculum", "mixture"])
    parser.add_argument("--include-power", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-division", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--use-run-plan", action="store_true",
                        help="Read active conditions from OUT_DIR/run_plan.json when present")
    parser.add_argument("--skip-complete", action="store_true",
                        help="Reuse matching full-budget final reports without reopening finished runs")
    add_tracking_arguments(parser)
    args = parser.parse_args()
    if args.use_run_plan:
        args.conditions = active_conditions(args.out_dir, args.conditions)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error("Seeds and conditions must not contain duplicates")
    config = Config.load(args.config)
    for name in ("include_power", "include_division"):
        value = getattr(args, name)
        if value is not None:
            setattr(config, name, value)
    directories = []
    for seed in args.seeds:
        for condition in args.conditions:
            directory = Path(args.out_dir) / f"{condition}-seed{seed}"
            run_config = replace(config, seed=seed, condition=condition)
            if args.skip_complete and completed_run(directory, run_config):
                print(f"Reusing completed {condition}-seed{seed}", flush=True)
                directories.append(directory)
                continue
            run(run_config, directory,
                resume=args.resume and (directory / "checkpoint.pt").exists(),
                tracking=tracking_arguments(args), final_evaluation=True)
            directories.append(directory)
    summary = compare(directories, Path(args.out_dir) / "comparison")
    for condition, result in summary["conditions"].items():
        print(f"{condition}: final target loss={result['mean_loss']:.4f}")


if __name__ == "__main__":
    main()
