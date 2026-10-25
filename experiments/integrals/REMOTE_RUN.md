# Remote integral experiment runbook

The [proposal](REMOTE_EXPERIMENT.md) uses 64 layers, width 512, 8 heads, and FF
width 2048 on one H100 80 GB. The commands below prepare data, calibrate, and run
the comparison. They do not provision a machine. H100 measurements and the final
main-run budget remain launch-time work.

## Environment

Use the same source revision and pinned numerical dependencies on preparation
and training hosts. Install `experiments/integrals/requirements.txt` into the
virtualenv. The GPU host needs CUDA-enabled PyTorch and a compatible driver;
runtime setup checks CUDA and BF16 support. Export uses Unix processes, signals,
and filesystem locks. Configure W&B authentication on the training host separately;
credentials are not stored in configurations or artifacts.

`configs/gpu_64layer.json` is **benchmark input**, not a calibrated training budget.
Its provisional steps and batch/context are replaced by benchmark output.

## 1. CPU pilot

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.integrals.corpus \
  --config experiments/integrals/configs/gpu_64layer.json \
  --kind pilot --pilot-examples 2048 --workers 32 \
  --out-dir out/integral-pilot
```

Benchmark worker count against available CPU cores/RAM. Each process owns its
numerical contexts and deadlines; work in flight is bounded. The pilot includes
all gamma levels, exact analytic maximum-length stress expressions, monitoring
banks, and separate baseline-calibration banks. Pilot expressions never become
optimizer-training samples. `preparation_metrics.json` reports new accepted
records, rejection counts, elapsed time, and throughput; numeric-label throughput
excludes the synthetic stress records.

Rerun the same command to resume. Only checksum-verified complete shards are reused.
A different recipe needs a new directory. Keep numerical policy and timeouts fixed.

## 2. GPU calibration

Transfer the complete pilot and code to the GPU host:

```bash
.venv/bin/python -m experiments.integrals.benchmark \
  --config experiments/integrals/configs/gpu_64layer.json \
  --data-dir out/integral-pilot --out-dir out/integral-calibration \
  --batches 64 128 256 512 1024 --contexts 128 256 \
  --ramp-minutes 50 --plateau-minutes 50
```

Candidates must pass several gamma probes and 86-token maximum-length records
with 10% VRAM headroom. The smallest batch within 5% of the best measured throughput
is selected and retested. The benchmark then measures every gamma and the actual
loss/decoding evaluations. Packing/transfer and optimizer computation are timed
separately. Benchmark weights are discarded.

`benchmark.json` records probes, failures, memory, useful-token throughput,
evaluation timing, source/environment metadata, and pilot manifest hash.
`config.json` freezes update count, explicit ramp length, batch/context, warmup,
checkpoint interval, and separate loss/decoding intervals. Each phase targets
50 minutes of optimizer computation; loading, evaluation, archiving, and
interruptions add wall time. All three conditions use the same update count.
LR warms up to 0.0003 and then stays constant.

`--allow-cpu` exists only for pipeline tests: its report explicitly records
`gpu_calibration: false`. CPU timings must not be used to budget an H100 run.
Release the benchmark GPU during full data preparation if it has no other work.

## 3. Complete CPU corpus

Copy the calibrated config back to the preparation host:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m experiments.integrals.corpus \
  --config out/integral-calibration/config.json \
  --workers 32 --tasks-per-shard 16 --out-dir out/integral-training-data
```

This exports curriculum and independent hard-only training batches, fixed
monitoring banks, fresh final banks, and separate baseline samples. Shuffled
training reuses exactly the curriculum batches. Hard-only has a separate
deterministic seed stream, including during the curriculum plateau. This is data
version `integral-full-sequence-v2-independent-hard`; earlier v1 smoke artifacts
are historical and cannot be mixed into the new recipe.

The training volume is 2*U*B accepted examples, plus evaluation/calibration banks.
The exporter does not repeat epochs to fill the budget. Shards contain raw token
bytes, memory-mapped offsets, metadata, and compressed numerical provenance.
Each shard is committed atomically; `manifest.json` appears only when the full
plan is complete. It records batch identities and file checksums. Keep an
independent durable copy of the corpus before the GPU runs.

Training verifies touched shards once per process and prefetches one batch.
The sweep verifies the full corpus before starting. Missing or corrupt data fails
closed: CUDA training never falls back to numerical integration.

## 4. Sequential comparison and persistence

Transfer the corpus and frozen config to the GPU host, then:

```bash
.venv/bin/python -m experiments.integrals.sweep \
  --config out/integral-calibration/config.json --data-dir /mnt/integral-data \
  --out-dir /mnt/work/integral-runs --archive-dir /mnt/durable/integral-archive \
  --conditions curriculum shuffled hard \
  --wandb online --wandb-project numerical-integrals-curriculum
```

Select a user/team with `--wandb-entity ENTITY` and optionally set `--wandb-group`.
The default group is the output directory name. All conditions use the same
paired model seed and independently initialize the same weights.

The archive must be a separately mounted durable filesystem that survives VM
deletion. Distinct paths alone do not establish cloud-storage durability.
At checkpoints, complete snapshots are copied and checksum-verified before an
atomic `current.json` pointer changes. The last two snapshots per condition are
retained, so failure during the next copy preserves an earlier recovery point.
Final snapshots also include local W&B files. Completion is marked only after
the W&B SDK finish/flush returns.

`--shutdown-on-success` explicitly enables `sudo -n shutdown -h now` after every
selected condition completes and all final archives and summaries verify. It
requires CUDA, the persistent archive, and host permission for that command.
Shutdown is disabled by default. Preparation and benchmarking never shut down,
delete, or provision resources. Persistent failure leaves diagnostics for review.

## 5. Resume and service operation

Rerun with `--resume` and the original config/corpus/conditions. Completed
conditions are verified and skipped. Partial ones restore model, optimizer, RNG,
step, immutable data identity, and metric-log position. The batch plan determines
the next shard/sample position. Changed manifests/recipes are rejected. The latest
two local checkpoints are retained in addition to the archive snapshots.

After loss of a working disk, read `<archive>/<condition>-seed0/current.json`,
verify the named snapshot, and copy its contents into the corresponding run
directory. Restore the archive's `run_plan.json` to the run root. The corpus may
move to a different path but must retain its manifest hash. Reuse the original
W&B project/entity/group and run ID. Interrupted offline runs can resume with
`--wandb disabled`; the SDK cannot reliably append to offline runs. Checkpoints
are trusted local artifacts, not safe inputs from untrusted sources.

`systemd/integral-runner.service` provides bounded restarts. Adapt its service
user and checkout path to the dedicated VM. Its environment file can contain:

```text
INTEGRAL_CONFIG=/opt/nanoGPT/out/integral-calibration/config.json
INTEGRAL_DATA=/mnt/integral-data
INTEGRAL_OUTPUT=/mnt/work/integral-runs
INTEGRAL_ARCHIVE=/mnt/durable/integral-archive
INTEGRAL_WANDB=online
INTEGRAL_SHUTDOWN_ON_SUCCESS=1
```

It invokes `remote_run.sh` with `--resume`. Installing/enabling the service,
mounting storage, and authorizing shutdown are explicit host setup steps.

## Metrics and validation

Every gamma 1..20 has independent procedural train/test banks. Loss and decoding
have separate intervals, with forced evaluation at the ramp boundary and endpoint.
Free decoding batches compatible lengths across the whole grid and never receives
gold answer bits. Final decoding count is independently configurable.

The six NLL metrics and numerical metrics retain their [definitions](README.md).
Independent calibration banks provide smoothed bit-prior NLL and always-zero
accuracy baselines. Local/final reports also include answer-NLL slices for
presence/absence of x, quantized zero/nonzero, and presence/absence of rounding
operators. These diagnostics do not alter the primary procedural distribution.

`summary.json` compares only complete fresh final evaluations with matching config
and corpus. It reports gamma-20 metrics and curriculum-minus-control answer-NLL
differences. One seed does not establish robustness across training seeds.

```bash
.venv/bin/python -m unittest \
  experiments.integrals.tests.test_integrals experiments.integrals.tests.test_remote \
  experiments.proposals.test_integral_ast_preview experiments.proposals.test_integral_distribution -v
bash -n experiments/integrals/remote_run.sh
```

CPU end-to-end validation exercises two-process label export, CPU-only calibration,
all three conditions from the resulting corpus, offline W&B, and durable snapshots.
Actual H100 VRAM, throughput, and CUDA resume require the planned remote preflight.
