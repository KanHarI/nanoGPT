"""Monitor the dedicated GPU sweep; dry-run unless --apply is supplied.

Uses only local reports and systemd. It never changes training configuration or
creates cloud resources. A stopped/evicted VM cannot run this monitor.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time

from .run_plan import CONDITIONS, active_conditions

TRAINING_UNITS = ("finite-field-training.service", "finite-field-runner.service")


def inspect_runs(directory):
    conditions = active_conditions(directory)
    runs = []
    for condition in conditions:
        path = directory / f"{condition}-seed0"
        result = dict(condition=condition, complete=False, step=0, gamma=None)
        config_path, final_path = path / "config.json", path / "final.json"
        metrics_path = path / "metrics.jsonl"
        try:
            if config_path.exists():
                config = json.loads(config_path.read_text())
                result["total_steps"] = config["steps"]
                if final_path.exists():
                    final = json.loads(final_path.read_text())
                    result["complete"] = (final["step"] == config["steps"]
                                          and final["config"] == config
                                          and config["condition"] == condition
                                          and config["seed"] == 0)
            if metrics_path.exists():
                # A concurrent write can leave a partial final line; ignore it.
                with metrics_path.open("rb") as handle:
                    handle.seek(max(0, metrics_path.stat().st_size - 131072))
                    lines = handle.read().decode("utf-8", errors="replace").splitlines()
                for line in reversed(lines):
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if row.get("kind") == "train":
                        result.update(step=row["step"], gamma=row["gamma"])
                        break
                result["last_progress_unix"] = metrics_path.stat().st_mtime
            elif config_path.exists():
                result["last_progress_unix"] = config_path.stat().st_mtime
        except (OSError, ValueError, KeyError, TypeError) as error:
            result["inspection_error"] = str(error)
        runs.append(result)
    comparison = directory / "comparison" / "summary.json"
    comparison_complete = False
    if comparison.exists():
        try:
            summary = json.loads(comparison.read_text())
            comparison_complete = (set(summary["conditions"]) == set(conditions)
                                   and summary["seeds"] == [0])
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return dict(runs=runs, all_complete=all(r["complete"] for r in runs),
                comparison_complete=comparison_complete)


def decide(snapshot, active, retries, now, previous=None, max_retries=2, stale_seconds=1800):
    previous = previous or {}
    progress = max((r.get("last_progress_unix", 0) for r in snapshot["runs"]), default=0)
    # Give an initial launch or recovery 30 minutes to produce progress.
    progress = max(progress, previous.get("last_restart_unix", 0),
                   previous.get("first_check_unix", now if not progress else 0))
    if active and now - progress < stale_seconds:
        return "monitor", "Training is active and progress is recent"
    if not active and snapshot["all_complete"] and snapshot["comparison_complete"]:
        return "shutdown_complete", "All planned final reports and the comparison are complete"
    if retries >= max_retries:
        return "shutdown_failed", "Recovery limit reached; preserve logs/checkpoints and stop billing compute"
    return "resume", "No active trainer" if not active else "No progress for at least 30 minutes"


def command(*args, check=True):
    return subprocess.run(args, capture_output=True, text=True, check=check, timeout=90)


def check_once(directory, apply=False):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "watchdog.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = directory / "watchdog.json"
        previous = json.loads(state_path.read_text()) if state_path.exists() else {}
        states = {unit: command("systemctl", "is-active", unit, check=False).stdout.strip()
                  for unit in TRAINING_UNITS}
        active_units = [unit for unit, state in states.items()
                        if state in ("active", "activating", "reloading")]
        now = time.time()
        snapshot = inspect_runs(directory)
        retries = previous.get("retries", 0)
        action, reason = decide(snapshot, bool(active_units), retries, now, previous)
        result = dict(timestamp=datetime.now(timezone.utc).isoformat(), action=action,
                      reason=reason, retries=retries, units=states,
                      first_check_unix=previous.get("first_check_unix", now), **snapshot)
        if "last_restart_unix" in previous:
            result["last_restart_unix"] = previous["last_restart_unix"]
        if not apply:
            print(json.dumps(result), flush=True)
            return result
        if action == "resume":
            result.update(retries=retries + 1, last_restart_unix=now)
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, indent=2) + "\n")
        temporary.replace(state_path)
        with (directory / "watchdog.jsonl").open("a") as handle:
            handle.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
        if action == "resume":
            for unit in (unit for unit, state in states.items()
                         if unit in active_units or state == "deactivating"):
                command("sudo", "systemctl", "stop", unit)
            # The sweep wrapper schedules shutdown at exit. Cancel it only for
            # a bounded recovery, then resume using its original frozen config.
            command("sudo", "shutdown", "-c", check=False)
            command("sudo", "systemctl", "reset-failed", "finite-field-runner.service", check=False)
            command("sudo", "systemctl", "start", "finite-field-runner.service")
        elif action.startswith("shutdown_"):
            command("sudo", "systemctl", "disable", "--now", "finite-field-watch.timer")
            command("sudo", "shutdown", "-h", "+2")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="out/gpu-64layer-gamma12")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    check_once(Path(args.out_dir), apply=args.apply)


if __name__ == "__main__":
    main()
