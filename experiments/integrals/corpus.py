"""Immutable, checksummed, memory-mapped procedural corpora and CPU export.

Preparation commits one shard at a time. Interrupted exports resume only from
verified complete shards. Training never falls back to numerical integration.
"""

import argparse
from collections import Counter, OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .config import Config, batch_plan, ramp_updates
from .data import Dataset, Example, STOI, VOCAB, VERSION, example_plan, stream_seed
from .encoding import encode

FORMAT = "integral-shards-v1"
FILES = ("tokens.bin", "offsets.npy", "metadata.jsonl", "metadata_offsets.npy", "provenance.jsonl.gz")


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b""):
            value.update(chunk)
    return value.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def data_recipe(config):
    keys = ("version", "seed", "gamma_low", "gamma_high", "max_operators", "steps", "batch_size",
            "eval_seed", "eval_gammas", "eval_examples", "final_examples", "baseline_examples", "label_timeout_seconds",
            "label_max_attempts")
    recipe = {**{key:config.to_dict()[key] for key in keys}, "ramp_updates":ramp_updates(config)}
    if config.training_gamma_sampling != "fixed":
        recipe["training_gamma_sampling"] = config.training_gamma_sampling
    if config.gamma_stage_steps:
        recipe["gamma_stage_steps"] = list(config.gamma_stage_steps)
    return recipe


def task_identity(entry):
    identity = {k:entry[k] for k in ("name", "gamma", "seed", "count")}
    if entry.get("gamma_sampling", "fixed") != "fixed":
        identity["gamma_sampling"] = entry["gamma_sampling"]
    return identity


def training_streams(conditions=None):
    """Shuffled and curriculum share a stream; hard has independent samples."""
    if conditions is None:
        return ["curriculum", "hard"]
    if not conditions or any(c not in ("curriculum", "shuffled", "hard") for c in conditions):
        raise ValueError("Invalid training conditions")
    selected = {"curriculum" if c == "shuffled" else c for c in conditions}
    return [c for c in ("curriculum", "hard") if c in selected]


def tasks(config, kind="training", pilot_examples=2048, conditions=None):
    result = []
    def add(name, gamma, seed, count, gamma_sampling="fixed"):
        task = dict(name=name, gamma=float(gamma), seed=seed, count=count)
        if gamma_sampling != "fixed":
            task["gamma_sampling"] = gamma_sampling
        result.append(task)
    if kind == "pilot":
        for gamma in dict.fromkeys([*config.eval_gammas, config.gamma_low, config.gamma_high]):
            mode = config.training_gamma_sampling if config.condition != "hard" and gamma >= 1 else "fixed"
            add(f"pilot/{gamma:g}", gamma, stream_seed(config.eval_seed, "pilot", f"{gamma:g}"), pilot_examples, mode)
        add("pilot/stress", config.gamma_high, stream_seed(config.eval_seed, "pilot-stress"), pilot_examples)
    else:
        for condition in training_streams(conditions):
            for spec in batch_plan(replace(config, condition=condition)):
                add(f"training/{condition}/{spec.index}", spec.gamma, spec.seed, config.batch_size, spec.gamma_sampling)
    for phase, count in (("monitor", config.eval_examples), ("final", config.final_examples)):
        if kind == "pilot" and phase == "final":
            continue
        for gamma in dict.fromkeys([*config.eval_gammas, config.gamma_high]):
            for split in ("train", "test"):
                key = f"{gamma:g}"
                add(f"eval/{phase}/{split}/{key}", gamma,
                    stream_seed(config.eval_seed, phase, key, split), count)
    for gamma in dict.fromkeys([*config.eval_gammas, config.gamma_high]):
        key = f"{gamma:g}"
        add(f"baseline/{key}", gamma, stream_seed(config.eval_seed, "baseline", key), config.baseline_examples)
    return result


def _generate(payload):
    task, settings = payload
    if task["name"] == "pilot/stress":
        k = settings["max_operators"]
        # Exact analytic integral: 1+...+1 on [0,1], with maximal binary length.
        prefix = ("0", "1", "1", *(("1", "+")*k), "INT_X", "=")
        example = Example(prefix, encode(k+1).tokens, str(k+1), "0", task["gamma"], k)
        return task, [(example.to_dict(), {"method":"analytic-benchmark-only", "value":k+1})]*task["count"], {}
    dataset = Dataset(None, **settings)
    output = []
    for gamma, seed in example_plan(task["gamma"], task["seed"], task["count"], task.get("gamma_sampling", "fixed")):
        example = dataset.sample(gamma, seed)
        output.append((example.to_dict(), dataset.last_provenance))
        if task["name"].startswith("training/"):
            dataset.stats[f"training_sample_gamma/{gamma:g}"] += 1
    return task, output, dict(dataset.stats)


def _bounded_map(executor, function, values, window):
    values = iter(values)
    pending = deque()
    for _ in range(window):
        try:
            pending.append(executor.submit(function, next(values)))
        except StopIteration:
            break
    while pending:
        result = pending.popleft().result()
        yield result
        try:
            pending.append(executor.submit(function, next(values)))
        except StopIteration:
            pass


def verify_shard(directory, info):
    if set(info["files"]) != set(FILES):
        raise ValueError("Unexpected shard file inventory")
    for name, expected in info["files"].items():
        if digest(Path(directory)/name) != expected:
            raise ValueError(f"Corrupt shard file: {directory}/{name}")


def _write_shard(directory, rows, number):
    name = f"part-{number:06d}"
    temporary = directory/(name+".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)  # Only this export's incomplete staging shard.
    temporary.mkdir()
    offsets, metadata_offsets, entries, counts = [0], [0], [], Counter()
    with (temporary/"tokens.bin").open("wb") as token_file, \
            (temporary/"metadata.jsonl").open("wb") as meta_file, \
            gzip.open(temporary/"provenance.jsonl.gz", "wt") as provenance:
        for task, examples, stats in rows:
            entries.append({**task, "first":len(offsets)-1, "shard":name})
            counts.update(stats)
            for row, diagnostics in examples:
                example = Example.from_dict(row)  # Stable target/domain syntax validated before export.
                tokens = bytes(STOI[t] for t in example.tokens)
                metadata = json.dumps({k:v for k,v in row.items() if k not in ("prefix", "answer")},
                                      separators=(",", ":"), allow_nan=False).encode()+b"\n"
                token_file.write(tokens)
                meta_file.write(metadata)
                offsets.append(offsets[-1]+len(tokens))
                metadata_offsets.append(metadata_offsets[-1]+len(metadata))
                provenance.write(json.dumps(diagnostics, separators=(",", ":"), allow_nan=False)+"\n")
    np.save(temporary/"offsets.npy", np.array(offsets, dtype=np.int64), allow_pickle=False)
    np.save(temporary/"metadata_offsets.npy", np.array(metadata_offsets, dtype=np.int64), allow_pickle=False)
    for filename in FILES:
        with (temporary/filename).open("rb") as handle:
            os.fsync(handle.fileno())
    info = dict(name=name, entries=entries, stats=dict(counts),
                files={filename:digest(temporary/filename) for filename in FILES})
    write_json(temporary/"shard.json", info)
    temporary.rename(directory/name)
    return info


def prepare(config, directory, workers=1, tasks_per_shard=16, kind="training", pilot_examples=2048,
            conditions=None):
    config.validate()
    if workers < 1 or tasks_per_shard < 1 or pilot_examples < 1 or kind not in ("training", "pilot"):
        raise ValueError("Invalid preparation settings")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    streams = training_streams(conditions)
    if kind == "pilot" and conditions is not None:
        raise ValueError("Condition selection applies only to training corpora")
    requested = tasks(config, kind, pilot_examples, streams)
    identity = dict(format=FORMAT, vocabulary=list(VOCAB), recipe=data_recipe(config), kind=kind,
                    pilot_examples=pilot_examples if kind == "pilot" else None, tasks_per_shard=tasks_per_shard)
    if streams != training_streams():
        identity["training_conditions"] = streams
    state_path = directory/"preparation.json"
    # One exporter per directory; flock is released automatically after a crash.
    import fcntl
    with (directory/".prepare.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_path.exists():
            if json.loads(state_path.read_text()) != identity:
                raise ValueError("Preparation directory belongs to a different data recipe")
        elif any(p.name != ".prepare.lock" for p in directory.iterdir()):
            raise FileExistsError("Refusing a nonempty, unidentified preparation directory")
        else:
            write_json(state_path, identity)
        shards, completed = [], []
        for folder in sorted(directory.glob("part-*")):
            if folder.name.endswith(".tmp"):
                continue
            info = json.loads((folder/"shard.json").read_text())
            if info["name"] != f"part-{len(shards):06d}" or folder.name != info["name"]:
                raise ValueError("Noncontiguous shard sequence")
            verify_shard(folder, info)
            shards.append(info)
            completed.extend(task_identity(e) for e in info["entries"])
        if completed != requested[:len(completed)]:
            raise ValueError("Prepared task identities do not match the requested plan")
        if (directory/"manifest.json").exists():
            manifest = json.loads((directory/"manifest.json").read_text())
            if len(completed) != len(requested) or manifest != {**identity, "shards":shards}:
                raise ValueError("Completed manifest mismatch")
            return directory/"manifest.json"
        settings = dict(max_operators=config.max_operators, timeout_seconds=config.label_timeout_seconds,
                        max_attempts=config.label_max_attempts)
        payloads = ((task, settings) for task in requested[len(completed):])
        started, buffer, new_examples, new_stats = time.perf_counter(), [], 0, Counter()
        executor = ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) if workers > 1 else None
        try:
            results = _bounded_map(executor, _generate, payloads, 2*workers) if executor else map(_generate, payloads)
            for result in results:
                buffer.append(result)
                new_examples += len(result[1])
                new_stats.update(result[2])
                if len(buffer) == tasks_per_shard:
                    shards.append(_write_shard(directory, buffer, len(shards)))
                    buffer = []
                    print(f"Prepared {sum(len(s['entries']) for s in shards)}/{len(requested)} batches/banks "
                          f"in {time.perf_counter()-started:.1f}s", flush=True)
            if buffer:
                shards.append(_write_shard(directory, buffer, len(shards)))
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
        write_json(directory/"manifest.json", {**identity, "shards":shards})
        elapsed = time.perf_counter()-started
        write_json(directory/"preparation_metrics.json", dict(workers=workers, new_examples=new_examples,
                   elapsed_seconds=elapsed, examples_per_second=new_examples/elapsed,
                   generated_labels_per_second=new_stats["generated"]/elapsed,
                   stats=dict(new_stats), resumed_completed_tasks=len(completed)))
    return directory/"manifest.json"


@dataclass(frozen=True)
class PreparedExample:
    tokens: tuple
    value: str
    uncertainty: str
    gamma: float
    operators: int
    attempts: int = 1

    @property
    def prefix(self):
        return self.tokens[:-17]

    @property
    def answer(self):
        return self.tokens[-17:-1]

    @property
    def text(self):
        return " ".join(self.tokens[:-1])+"\n"


class Corpus:
    def __init__(self, directory, config=None, require_training=True, verify_all=False):
        self.directory = Path(directory)
        path = self.directory/"manifest.json"
        self.manifest = json.loads(path.read_text())
        self.fingerprint = digest(path)
        if self.manifest["format"] != FORMAT or self.manifest["vocabulary"] != list(VOCAB):
            raise ValueError("Unsupported corpus format or vocabulary")
        if require_training and self.manifest["kind"] != "training":
            raise ValueError("Pilot corpus cannot be used for training")
        if config and self.manifest["recipe"] != data_recipe(config):
            raise ValueError("Corpus does not match the configured data plan")
        self.shards, self.entries, self.by_identity = {}, {}, {}
        self.verified, self.maps, self.stats = set(), OrderedDict(), Counter()
        for i, shard in enumerate(self.manifest["shards"]):
            name = shard["name"]
            if name != f"part-{i:06d}":
                raise ValueError("Invalid shard path/sequence")
            self.shards[name] = shard
            for entry in shard["entries"]:
                if entry["name"] in self.entries or entry["shard"] != name:
                    raise ValueError("Duplicate/misplaced corpus entry")
                self.entries[entry["name"]] = entry
                identity = (float(entry["gamma"]), entry["seed"], entry["count"], entry.get("gamma_sampling", "fixed"))
                if identity in self.by_identity:
                    raise ValueError("Duplicate sample identity")
                self.by_identity[identity] = entry
            if verify_all:
                self._verify(name)
        if config:
            streams = training_streams(self.manifest.get("training_conditions"))
            required = "curriculum" if config.condition == "shuffled" else config.condition
            if require_training and required not in streams:
                raise ValueError("Corpus does not contain the requested training condition")
            requested = tasks(config, self.manifest["kind"], self.manifest.get("pilot_examples") or 2048, streams)
            actual = [task_identity(e) for e in self.entries.values()]
            if actual != requested:
                raise ValueError("Incomplete or changed corpus task inventory")

    def _verify(self, name):
        if name not in self.verified:
            verify_shard(self.directory/name, self.shards[name])
            self.verified.add(name)

    def _load(self, name):
        self._verify(name)
        if name not in self.maps:
            folder = self.directory/name
            self.maps[name] = (np.memmap(folder/"tokens.bin", mode="r", dtype=np.uint8),
                               np.load(folder/"offsets.npy", mmap_mode="r", allow_pickle=False),
                               np.memmap(folder/"metadata.jsonl", mode="r", dtype=np.uint8),
                               np.load(folder/"metadata_offsets.npy", mmap_mode="r", allow_pickle=False))
            if len(self.maps) > 4:
                self.maps.popitem(last=False)
        self.maps.move_to_end(name)
        return self.maps[name]

    def read(self, entry):
        tokens, offsets, metadata, meta_offsets = self._load(entry["shard"])
        output = []
        for index in range(entry["first"], entry["first"]+entry["count"]):
            row = json.loads(metadata[meta_offsets[index]:meta_offsets[index+1]].tobytes())
            output.append(PreparedExample(tuple(VOCAB[t] for t in tokens[offsets[index]:offsets[index+1]]), **row))
        self.stats["loaded_examples"] += len(output)
        return output

    def named(self, name):
        return self.read(self.entries[name])

    def examples(self, gamma, seed, count, gamma_sampling="fixed"):
        try:
            entry = self.by_identity[float(gamma), seed, count, gamma_sampling]
        except KeyError as exc:
            raise ValueError("Missing prepared batch/bank; numerical generation is disabled") from exc
        return self.read(entry)

    def training_examples(self, spec, count):
        return self.examples(spec.gamma, spec.seed, count, spec.gamma_sampling)


class Prefetch:
    """One lookahead batch; no RNG or numerical integration in the IO thread."""
    def __init__(self, dataset, specs, count):
        self.dataset, self.specs, self.count = dataset, specs, count

    def __iter__(self):
        with ThreadPoolExecutor(max_workers=1) as pool:
            def fetch(spec):
                return spec, self.dataset.training_examples(spec, self.count)
            yield from _bounded_map(pool, fetch, self.specs, 2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--tasks-per-shard", type=int, default=16)
    parser.add_argument("--kind", choices=("training", "pilot"), default="training")
    parser.add_argument("--pilot-examples", type=int, default=2048)
    parser.add_argument("--conditions", nargs="+", choices=("curriculum", "shuffled", "hard"),
                        help="Export only selected training streams; default exports both")
    args = parser.parse_args()
    print(prepare(Config.load(args.config), args.out_dir, args.workers, args.tasks_per_shard,
                  args.kind, args.pilot_examples, args.conditions))


if __name__ == "__main__":
    main()
