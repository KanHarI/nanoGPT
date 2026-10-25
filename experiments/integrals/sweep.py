"""Run matched conditions, archive verified checkpoints, and optionally shut down.

The archive must be a separately mounted durable filesystem. This module does
not create cloud resources or infer that an arbitrary directory is persistent.
"""

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

import torch

from .config import Config
from .corpus import Corpus, digest, write_json
from .tracking import add_arguments, from_arguments
from .train import run
from .data import stream_seed

CONDITIONS = ("curriculum", "shuffled", "hard")


def shutdown_host():
    subprocess.run(["sudo", "-n", "shutdown", "-h", "now"], check=True)


class Archive:
    """Atomic snapshot pointer; retain two recoverable snapshots per condition."""
    def __init__(self, directory, source):
        self.directory, self.source = Path(directory).resolve(), Path(source).resolve()
        if self.directory.is_relative_to(self.source) or self.source.is_relative_to(self.directory):
            raise ValueError("Archive and run directories must be separate, nonnested paths")
        self.directory.mkdir(parents=True, exist_ok=True)

    def snapshot(self, source, include_wandb=False):
        source = Path(source)
        destination = self.directory/source.name
        destination.mkdir(exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".snapshot-", dir=destination))
        files = {}
        try:
            paths = [p for p in source.iterdir() if p.is_file() and p.suffix in (".json", ".jsonl", ".pt")]
            if include_wandb and (source/"wandb").exists():
                paths += [p for p in (source/"wandb").rglob("*") if p.is_file() and not p.is_symlink()]
            for path in paths:
                # The previous checkpoint is already retained in the previous snapshot.
                if path.name in ("checkpoint-previous.pt", "receipt.json"):
                    continue
                relative = path.relative_to(source)
                target = temporary/relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                expected = digest(path)
                if digest(target) != expected:
                    raise IOError(f"Archive checksum mismatch: {relative}")
                with target.open("rb") as handle:
                    import os
                    os.fsync(handle.fileno())
                files[str(relative)] = expected
            if not {"checkpoint.pt", "config.json", "metrics.jsonl", "wandb_history.jsonl"} <= files.keys():
                raise ValueError("Incomplete checkpoint snapshot")
            checkpoint = torch.load(temporary/"checkpoint.pt", map_location="cpu", weights_only=False)
            for filename, key in (("metrics.jsonl", "records"), ("wandb_history.jsonl", "history_records")):
                if len((temporary/filename).read_text().splitlines()) != checkpoint[key]:
                    raise ValueError("Archived checkpoint/log positions disagree")
            name = f"snapshot-{checkpoint['step']:09d}-{temporary.name.removeprefix('.snapshot-')}"
            receipt = dict(snapshot=name, step=checkpoint["step"], completed=checkpoint.get("completed", False), files=files)
            write_json(temporary/"receipt.json", receipt)
            temporary.rename(destination/name)
            write_json(destination/"current.json", receipt)
            snapshots = sorted(destination.glob("snapshot-*"), key=lambda p:p.stat().st_mtime_ns, reverse=True)
            for old in snapshots[2:]:
                shutil.rmtree(old)
            return destination/name
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def verify(self, name, completed=True):
        destination = self.directory/name
        receipt = json.loads((destination/"current.json").read_text())
        folder = receipt["snapshot"]
        if not folder.startswith("snapshot-") or Path(folder).name != folder:
            raise ValueError("Invalid snapshot path")
        if completed and not receipt["completed"]:
            raise ValueError("Final checkpoint has not been archived")
        for filename, expected in receipt["files"].items():
            if Path(filename).is_absolute() or ".." in Path(filename).parts:
                raise ValueError("Invalid archived filename")
            if digest(destination/folder/filename) != expected:
                raise ValueError(f"Archived artifact failed verification: {filename}")
        return destination/folder


def completed_report(directory, config, fingerprint):
    directory = Path(directory)
    final = json.loads((directory/"final.json").read_text())
    checkpoint = torch.load(directory/"checkpoint.pt", map_location="cpu", weights_only=False)
    if (final.get("phase") != "final" or final["step"] != config.steps or final["config"] != config.to_dict()
            or final.get("data_fingerprint") != fingerprint or not checkpoint.get("completed")
            or checkpoint["step"] != config.steps or checkpoint["config"] != config.to_dict()
            or checkpoint.get("data_fingerprint") != fingerprint):
        raise ValueError("Run is not a complete result for this exact recipe and corpus")
    expected = {f"{g:g}" for g in [*config.eval_gammas, config.gamma_high]}
    for split in ("train", "test"):
        if set(final["loss_grid"][split]) != expected:
            raise ValueError("Incomplete final evaluation grid")
        for gamma,cell in final["loss_grid"][split].items():
            if (cell["examples"] != config.final_examples or
                    cell["seed"] != stream_seed(config.eval_seed, "final", gamma, split) or
                    cell["generation_examples"] != min(config.final_examples,
                        config.final_generation_examples or config.generation_examples)):
                raise ValueError("Wrong final sample count")
            for name in ("answer_nll", "full_nll", "non_answer_nll", "sign_nll", "exponent_nll", "mantissa_nll"):
                if not math.isfinite(cell[name]) or cell[name] < 0:
                    raise ValueError("Invalid final losses")
    return final


def sweep(config, data_dir, out_dir, conditions=CONDITIONS, resume=False, tracking=None,
          archive_dir=None, shutdown_on_success=False):
    config.validate()
    if not conditions or len(set(conditions)) != len(conditions) or set(conditions)-set(CONDITIONS):
        raise ValueError("Conditions must be unique and supported")
    if shutdown_on_success and (not archive_dir or config.device != "cuda"):
        raise ValueError("Shutdown requires CUDA and a separately configured persistent archive")
    corpus = Corpus(data_dir, config, verify_all=True)
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    archive = Archive(archive_dir, directory) if archive_dir else None
    identity = dict(config=config.to_dict(), conditions=list(conditions), data_fingerprint=corpus.fingerprint)
    plan_path = directory/"run_plan.json"
    import fcntl
    with (directory/".runner.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if plan_path.exists():
            if not resume or json.loads(plan_path.read_text()) != identity:
                raise ValueError("Use --resume with the original run plan and corpus")
        elif any(p.name != ".runner.lock" for p in directory.iterdir()):
            raise FileExistsError("Unidentified nonempty run directory")
        else:
            write_json(plan_path, identity)
        reports = {}
        for condition in conditions:
            current = replace(config, condition=condition)
            output = directory/f"{condition}-seed{config.seed}"
            complete = False
            if resume and (output/"final.json").exists() and (output/"checkpoint.pt").exists():
                try:
                    reports[condition] = completed_report(output, current, corpus.fingerprint)
                    complete = True
                except ValueError:
                    pass  # An incomplete endpoint is resumed from its committed checkpoint.
            if not complete:
                run(current, output, resume=resume and (output/"checkpoint.pt").exists(),
                    tracking=tracking, data_dir=data_dir,
                    checkpoint_callback=archive.snapshot if archive else None)
                reports[condition] = completed_report(output, current, corpus.fingerprint)
            if archive:
                archive.snapshot(output, include_wandb=True)
                archive.verify(output.name)
        target = f"{config.gamma_high:g}"
        summary = dict(config=config.to_dict(), data_fingerprint=corpus.fingerprint,
                       primary_gamma=config.gamma_high, paired_model_seed=config.seed,
                       conditions={c:r["loss_grid"]["test"][target] for c,r in reports.items()})
        if "curriculum" in reports:
            reference = reports["curriculum"]["loss_grid"]["test"][target]["answer_nll"]
            summary["curriculum_minus_control_answer_nll"] = {
                c:reference-r["loss_grid"]["test"][target]["answer_nll"] for c,r in reports.items() if c != "curriculum"}
        write_json(directory/"summary.json", summary)
        if archive:
            for name in ("run_plan.json", "summary.json"):
                write_json(archive.directory/name, json.loads((directory/name).read_text()))
                if digest(archive.directory/name) != digest(directory/name):
                    raise IOError("Archived run summary mismatch")
            for condition in conditions:
                archive.verify(f"{condition}-seed{config.seed}")
        write_json(directory/"complete.json", dict(status="complete", conditions=list(conditions),
                                                   data_fingerprint=corpus.fingerprint,
                                                   archive_verified=archive is not None))
        if shutdown_on_success:
            shutdown_host()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--conditions", choices=CONDITIONS, nargs="+", default=list(CONDITIONS))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--archive-dir", help="Separately mounted durable filesystem, not ephemeral VM storage")
    parser.add_argument("--shutdown-on-success", action="store_true")
    add_arguments(parser)
    args = parser.parse_args()
    print(sweep(Config.load(args.config), args.data_dir, args.out_dir, args.conditions, args.resume,
                from_arguments(args), args.archive_dir, args.shutdown_on_success))


if __name__ == "__main__":
    main()
