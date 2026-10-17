"""Optional W&B tracking and import of checkpoint-committed JSONL history.

Tracking options are separate from the experimental Config: enabling logging
must not change checkpoint compatibility or what counts as a matched trial.
"""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import uuid

from .config import Config
from .data import PRIMES


@dataclass
class Tracking:
    mode: str = "disabled"
    project: str = "finite-field-curriculum"
    entity: str | None = None
    group: str | None = None
    checkpoint: bool = False


def add_arguments(parser):
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb-project", default="finite-field-curriculum")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-checkpoint", action="store_true",
                        help="Upload the latest checkpoint as a model artifact at clean exit")


def from_arguments(args):
    return Tracking(args.wandb, args.wandb_project, args.wandb_entity,
                    args.wandb_group, args.wandb_checkpoint)


def flatten(values, prefix):
    """Only numeric values become chart metrics; metadata stays in config/files."""
    result = {}
    symbols = {"+": "add", "-": "subtract", "*": "multiply", "/": "divide", "^": "power"}
    for key, value in values.items():
        name = f"{prefix}/{symbols.get(str(key), str(key))}"
        if isinstance(value, dict):
            result.update(flatten(value, name))
        elif isinstance(value, (int, float)) and math.isfinite(value):
            result[name] = value
    return result


def metrics(record, loss_only=False):
    if loss_only:
        if record["kind"] != "loss_grid":
            return None
        return grid_metrics(record["loss_grid"], record["step"])
    prefix = "eval/high_gamma" if record["kind"] == "eval" else "train"
    values = {k: v for k, v in record.items() if k not in ("kind", "step", "seed")}
    result = flatten(values, prefix)
    result["optimizer_step"] = record["step"]
    result["baseline/uniform_loss"] = sum(math.log(p) for p in PRIMES) / len(PRIMES)
    result["baseline/uniform_accuracy"] = sum(1 / p for p in PRIMES) / len(PRIMES)
    if record["kind"] == "train":
        result["curriculum/gamma"] = record["gamma"]
    seconds = record.get("train_seconds", 0)
    if seconds > 0 and "supervised_answers" in record:
        result["throughput/answers_per_second"] = record["supervised_answers"] / seconds
    return result


def grid_metrics(grid, step):
    values = {"optimizer_step": step}
    for split, gammas in grid.items():
        for gamma, result in gammas.items():
            for prime, field in result["by_field"].items():
                values[f"eval/gamma_{float(gamma):02g}/field_{int(prime):02d}/{split}_loss"] = field["loss"]
    return values


class Tracker:
    def __init__(self, options, config, out_dir, resume=False, job_type="train"):
        self.options = options or Tracking()
        self.config, self.directory = config, Path(out_dir)
        self.resume, self.job_type = resume, job_type
        self.run = None

    def __enter__(self):
        if self.options.mode == "disabled":
            return self
        if self.options.mode not in ("online", "offline"):
            raise ValueError("Invalid W&B mode")
        try:
            import wandb
        except ImportError as error:
            raise RuntimeError("Install requirements-wandb.txt to enable W&B") from error
        self.sdk = wandb
        identity_path = self.directory / "wandb_run.json"
        identity = json.loads(identity_path.read_text()) if identity_path.exists() else {}
        if identity and not self.resume:
            raise ValueError("This directory already has a W&B run; resume it instead")
        entity = self.options.entity or identity.get("entity") or None
        group = self.options.group or identity.get("group") or self.directory.parent.name
        if identity and (identity["project"] != self.options.project
                         or (entity and identity.get("entity") and entity != identity["entity"])
                         or identity["group"] != group):
            raise ValueError("W&B destination/group cannot change when resuming a run")
        if identity and self.options.mode == "offline":
            raise ValueError("W&B offline history cannot be resumed; sync the offline run with "
                             "wandb sync, then resume with --wandb online, or disable tracking")
        self.run = wandb.init(
            project=self.options.project, entity=entity, group=group,
            id=identity.get("id") or uuid.uuid4().hex[:8],
            name=f"{self.config.condition}-seed{self.config.seed}",
            job_type=self.job_type, tags=["finite-field", self.config.condition,
                                         "power" if self.config.include_power else "no-power",
                                         "division" if self.config.include_division else "no-division"],
            config=self.config.to_dict(), mode=self.options.mode,
            resume=("must" if identity else "allow") if self.options.mode == "online" else None,
            dir=str(self.directory), save_code=False,
            settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                    x_disable_stats=self.config.eval_mode == "loss_grid", init_timeout=45),
        )
        try:
            identity = dict(id=self.run.id, project=self.run.project, entity=self.run.entity,
                            group=group, mode=self.options.mode,
                            url=self.run.url if self.options.mode == "online" else None)
            temporary = self.directory / "wandb_run.tmp"
            temporary.write_text(json.dumps(identity, indent=2) + "\n")
            temporary.replace(identity_path)
            self.run.define_metric("optimizer_step", hidden=self.config.eval_mode == "loss_grid")
            self.run.define_metric("*", step_metric="optimizer_step", step_sync=False)
        except BaseException:
            self.run.finish(exit_code=1)
            raise
        if identity["url"]:
            print(f"W&B: {identity['url']}", flush=True)
        return self

    def log(self, record):
        if self.run is not None:
            # Internal W&B events increase independently, so train/eval can have
            # the same optimizer step without dropping either record.
            values = metrics(record, loss_only=self.config.eval_mode == "loss_grid")
            if values is not None:
                self.run.log(values)

    def final(self, report):
        if self.run is None:
            return
        if "loss_grid" in report:
            values = grid_metrics(report["loss_grid"], report["step"])
            self.run.log(values)
            self.run.summary.update(values)
            return
        values = {"optimizer_step": report["step"]}
        for result in report["evaluations"]:
            values.update(flatten(result, f"final/gamma_{result['gamma']:g}"))
        self.run.log(values)
        self.run.summary.update(values)

    def finish_training(self, step, checkpoint_path):
        if self.run is None:
            return
        self.run.summary.update({"training/completed_updates": step,
                                 "training/status": "complete" if step == self.config.steps else "checkpointed"})
        artifact = self.sdk.Artifact(f"run-{self.run.id}-report", type="experiment-report")
        for name in ("config.json", "environment.json", "final.json"):
            path = self.directory / name
            if path.exists():
                if name == "final.json" and json.loads(path.read_text())["step"] != step:
                    continue
                artifact.add_file(str(path), name=name)
        self.run.log_artifact(artifact)
        if self.options.checkpoint:
            artifact = self.sdk.Artifact(f"run-{self.run.id}-checkpoint", type="model",
                                         metadata={"optimizer_step": step})
            artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
            self.run.log_artifact(artifact, aliases=["latest", f"step-{step}"])

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is not None:
            if exc_type is not None:
                self.run.summary["training/status"] = "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
            self.run.finish(exit_code=0 if exc_type is None else 1)
        return False


def import_history(directory, options):
    """Upload only records covered by the latest checkpoint, without training."""
    import torch

    directory = Path(directory)
    config = Config.load_saved(directory / "config.json")
    checkpoint_path = directory / "checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    lines = (directory / "metrics.jsonl").read_text().splitlines()
    if len(lines) < checkpoint["records"]:
        raise ValueError("Metrics log is shorter than its checkpoint")
    if options.mode == "disabled":
        raise ValueError("History import requires --wandb online or offline")
    with Tracker(options, config, directory, job_type="history-import") as tracker:
        for line in lines[:checkpoint["records"]]:
            tracker.log(json.loads(line))
        final = directory / "final.json"
        if final.exists():
            report = json.loads(final.read_text())
            if report["step"] == checkpoint["step"]:
                tracker.final(report)
        tracker.finish_training(checkpoint["step"], checkpoint_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-run", required=True)
    add_arguments(parser)
    args = parser.parse_args()
    import_history(args.import_run, from_arguments(args))


if __name__ == "__main__":
    main()
