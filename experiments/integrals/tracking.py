"""Real W&B SDK logging, with an inspectable local mirror in every mode."""

from dataclasses import dataclass
import json
from pathlib import Path
import uuid


@dataclass
class Tracking:
    mode: str = "disabled"
    project: str = "numerical-integrals-curriculum"
    entity: str | None = None
    group: str | None = None


def add_arguments(parser):
    parser.add_argument("--wandb", choices=("disabled", "offline", "online"), default="disabled")
    parser.add_argument("--wandb-project", default="numerical-integrals-curriculum")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")


def from_arguments(args):
    return Tracking(args.wandb, args.wandb_project, args.wandb_entity, args.wandb_group)


def grid_metrics(grid, step, prefix="eval"):
    values = {"optimizer_step": step}
    for split, results in grid.items():
        for gamma, result in results.items():
            for name, value in result.items():
                if name not in ("gamma", "seed") and value is not None and not isinstance(value, dict):
                    values[f"{prefix}/gamma_{float(gamma):02g}/{split}/{name}"] = value
    return values


class Tracker:
    def __init__(self, options, config, directory, resume=False):
        self.options = options or Tracking()
        self.config, self.directory, self.resume = config, Path(directory), resume
        self.run = None

    def __enter__(self):
        options = self.options
        if options.mode == "disabled":
            return self
        if options.mode not in ("offline", "online"):
            raise ValueError("Invalid W&B mode")
        import wandb
        self.sdk = wandb
        path = self.directory/"wandb_run.json"
        identity = json.loads(path.read_text()) if path.exists() else {}
        entity = options.entity or identity.get("entity")
        group = options.group or identity.get("group") or self.directory.parent.name
        if identity and (not self.resume or options.mode == "offline" or
                         identity["project"] != options.project or identity["group"] != group or
                         (entity and identity["entity"] != entity)):
            raise ValueError("Resume with original online W&B destination, or disable tracking")
        self.run = wandb.init(project=options.project, entity=entity, group=group,
                             id=identity.get("id") or uuid.uuid4().hex[:8],
                             name=f"{self.config.condition}-seed{self.config.seed}",
                             tags=["integrals", "full-sequence-nll", self.config.condition],
                             config=self.config.to_dict(), mode=options.mode, dir=str(self.directory),
                             resume=("must" if identity else "allow") if options.mode == "online" else None,
                             settings=wandb.Settings(console="off", disable_git=True, disable_code=True,
                                                     x_disable_stats=True, init_timeout=45, quiet=True))
        try:
            path.write_text(json.dumps(dict(id=self.run.id, project=self.run.project,
                                            entity=self.run.entity, group=group,
                                            url=self.run.url if options.mode == "online" else None), indent=2)+"\n")
            self.run.define_metric("optimizer_step", hidden=True)
            self.run.define_metric("*", step_metric="optimizer_step", step_sync=False)
        except BaseException:
            self.run.finish(exit_code=1)
            raise
        return self

    def log(self, values):
        # Also runs when the SDK is disabled. Mirrors exactly what is sent to W&B.
        with (self.directory/"wandb_history.jsonl").open("a") as handle:
            handle.write(json.dumps(values, allow_nan=False)+"\n")
        if self.run is not None:
            self.run.log(values)

    def finish_training(self, step, final=None):
        status = "complete" if step == self.config.steps else "checkpointed"
        if final:
            # Keep the fresh final bank separate from the fixed monitoring curves.
            self.log(grid_metrics(final["loss_grid"], step, prefix="final"))
        if self.run is None:
            return
        self.run.summary.update({"training/completed_updates": step, "training/status": status})
        artifact = self.sdk.Artifact(f"run-{self.run.id}-report", type="experiment-report")
        for name in ("config.json", "environment.json", "vocabulary.json", "dataset.json", *(("final.json",) if final else ())):
            artifact.add_file(str(self.directory/name), name=name)
        self.run.log_artifact(artifact)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.run is not None:
            if exc_type is not None:
                self.run.summary["training/status"] = "failed"
            self.run.finish(exit_code=0 if exc_type is None else 1)
        return False
