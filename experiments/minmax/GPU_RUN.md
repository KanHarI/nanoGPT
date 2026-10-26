# GPU min/max experiment

The GPU VMs and checkpoint disks were deleted on 2026-10-18, followed by project
shutdown. See [cloud cleanup](../CLOUD_CLEANUP.md). VM paths and retained-disk
statements below describe the experiment at the time it ran.

The gamma-12 comparison documented here is complete. The subsequent
[gamma-20 repeat](GPU_GAMMA20.md) uses a separate config, output directory,
and W&B group with the same optimizer-update budget.

The current experiment uses a **20-minute gamma ramp followed by 20 minutes
at gamma 12**, as requested on 2026-10-17. Curriculum, hard, and shuffled run
sequentially with one seed. The earlier longer-budget run
[e2389351](https://wandb.ai/kanhar/minmax-curriculum/runs/e2389351) was stopped
and marked superseded. Its checkpoint and original configuration remain in
`out/minmax-gpu-24layer-gamma12/`; the new comparison starts from fresh weights
in `out/minmax-gpu-24layer-gamma12-20m20m/`.
The preserved original checkpoint is at update 11,380, with 708.19 seconds of
training recorded. It is a partial exploratory run, excluded from the revised
matched comparison and the current workspace filter.
The revised [curriculum run](https://wandb.ai/kanhar/minmax-curriculum/runs/5a10dd7c)
started on 2026-10-17 at approximately 10:27 UTC.

## Calibrated recipe

The measured recipe is [gpu_calibrated.json](configs/gpu_calibrated.json):
**26,668 updates, 256 answers/update, context 256, 18,964,736 parameters**.
Gamma reaches 12 on update 15,288; the remaining 11,380 updates stay at 12.
Warmup lasts 955 updates, followed by constant LR 0.0006.
Each condition receives 6,827,008 supervised answers. The learning-rate warmup
remains 955 updates; the 20-minute ramp refers to gamma, not the learning rate.
The [schedule derivation](results/gpu_schedule_20m20m.json) records the original
and revised recipes. It uses `ceil(1200 / mean_ramp_update_seconds)` and
`ceil(1200 / gamma12_update_seconds)` for the two phases. The benchmark report
retains its original 50-minute schedule as historical measurement metadata.

The [benchmark report](results/gpu_benchmark.json) tested all twelve
batch/context combinations. The selected pair repeated at 2,448 answers/second
at gamma 12, with 2.32 GB allocated and 2.70 GB reserved GPU memory. Keeping one
expression per row achieved 405 answers/second at the same batch/context sizes.
The monitoring pass took 0.85 seconds, versus approximately 120 seconds of
gamma-12 training between evaluations (about 0.7% evaluation time, excluding
checkpoint writes and startup). Monitoring runs every 1,138 updates.

Calibration predicts about 40 minutes of training for curriculum and shuffled,
and 46.9 minutes for hard because it receives the same number of updates on
longer expressions throughout. Total training is about 2 hours 7 minutes, plus
evaluation, startup, final reports, and checkpoint writes. These are estimates; live timings are
recorded in each run's metrics. The ramp targets training time, including
procedural generation and packing, and excludes evaluation/checkpoint time.

The model has 24 layers, width 256, four attention heads, FF width 1024,
and a 48-token vocabulary. With a 128-token context it has 18,931,968
parameters; a 256-token context adds 32,768 position-embedding parameters.
Unary negation applies to entire subtrees. The generator retains all sampled
trees, including repeated min/max operators and double negations.

## Infrastructure

Project: `finite-field-curriculum-kanhar`. Dedicated VM: `minmax-l4` in
`us-central1-a`, a Spot `g2-standard-12` with one NVIDIA L4 (23,034 MiB as
reported by the driver), 12 vCPUs, and a retained 200 GB boot disk.
Python 3.12.3, PyTorch 2.9.1+cu129, NVIDIA driver 580.178.04, W&B 0.30.0.

The old `finite-field-h100` VM is stopped, with its disk/checkpoints retained.
Restarting it and requesting H100s in zones `us-central1-b` and `us-central1-c`
failed because of GPU capacity. The project has no A100 quota. The L4 is a
separate VM; no finite-field outputs are moved or overwritten.

The VM has a 12-hour maximum run duration with stop action and no disk deletion.
This includes setup and benchmarking time. The run monitor additionally stops
the VM after completion or after bounded recovery fails. Stopped disks still
incur storage charges. A Spot eviction can interrupt training; the in-VM monitor
cannot restart a stopped or evicted VM.

## Frozen protocol

Run one seed, 0, for `curriculum`, `hard`, and `shuffled`, in that order, starting
each model and optimizer from scratch. Use identical update and answer budgets.
Shuffled replays the exact curriculum batches in a different order. Hard uses
gamma 12 throughout. Curriculum increases gamma linearly from 1 to 12 over the
first 15,288 updates (about 57.3% of the total) and then stays at 12. The ramp
is linear in update number. It occupies more than half the updates because
lower-gamma updates are faster; the two phases have similar training duration.

The benchmark tests complete-expression packing at contexts 128 and 256,
with 64, 128, 256, 512, 1,024, and 2,048 supervised answers per update. It
includes fresh generation, packing, BF16 forward/backward, clipping, AdamW,
and answer statistics. Choose the smallest batch within 5% of the fastest
measured throughput, requiring at least 10% GPU-memory headroom. Repeat the
winner and measure each integer gamma. Reuse these timings to calibrate the
20-minute ramp and the 20-minute plateau separately.
This is a measured choice among tested candidates, not a global optimum.
Calibration models and optimizer state are discarded before training.

Each context starts at an expression boundary. Whole `expression=answer\n`
items share contexts when they fit; loss applies only to the single answer
token. No sample is split. Batch size counts answers, not packed rows.

AdamW uses LR 0.0006 after linear warmup, with no decay, betas (0.9, 0.95),
weight decay 0.1, and gradient clipping at 1. The original 955-update LR
warmup is retained. Monitoring uses 512 independent samples per gamma/split;
fresh final evaluation uses 4,096. Baselines use 4,096 separate calibration
samples per gamma. Evaluation is scheduled about every 120 seconds of
gamma-12 training; low-gamma updates can be faster. Actual evaluation duration
is recorded locally in `metrics.jsonl` as `eval_seconds`.

Only the 24 train/test loss curves for gamma 1–12 are logged to W&B. Both
banks are procedural evaluation data, independent of the optimizer samples.
The [W&B workspace](https://wandb.ai/kanhar/minmax-curriculum?nw=4xpnpd977nh)
contains twelve panels. Local reports also contain accuracy, structural
diagnostics, training timing, and baselines.

## Commands and outputs

```sh
# From the repository root on the VM, using the frozen 20m+20m config:
bash experiments/minmax/gpu_run.sh
```

The source deployment includes `source_snapshot.json`, recording the source
branch, base commit, dirty status, and file hashes. Each training run also
records source hashes in `environment.json`. The benchmark writes
`out/minmax-benchmark/benchmark.json` and its calibrated `config.json`.

Checkpoints are on the persistent boot disk, under:

```text
/home/kanhar/nanoGPT/out/minmax-gpu-24layer-gamma12-20m20m/
  curriculum-seed0/checkpoint.pt
  hard-seed0/checkpoint.pt
  shuffled-seed0/checkpoint.pt
```

Each run also contains `config.json`, `environment.json`, `metrics.jsonl`,
`wandb_run.json`, and, upon completion, `final.json`. The sweep writes
`summary.json` after all three fresh final evaluations finish. W&B receives
small configuration/final-report artifacts; checkpoints remain on the VM disk.

## Monitoring and recovery

Install the three units from [systemd](systemd/) on the dedicated VM, reload
systemd, start `minmax-runner.service`, then enable `minmax-watch.timer`.
The runner uses `--resume`, skips complete matching runs, and begins the next
condition automatically. The timer checks every 15 minutes and also checks
when the runner exits. A 30-minute lack of log progress triggers recovery.
At most two recovery attempts are allowed; final completion or exhausted
recovery schedules shutdown after two minutes.

```sh
gcloud compute ssh minmax-l4 --project=finite-field-curriculum-kanhar \
  --zone=us-central1-a
sudo systemctl status minmax-runner.service minmax-watch.timer
tail -n 20 /home/kanhar/nanoGPT/minmax-training.log
cat /home/kanhar/nanoGPT/out/minmax-gpu-24layer-gamma12-20m20m/watchdog.json
```

To intentionally stop training without the monitor restarting it, first
disable the timer and temporarily clear the runner's `ExecStopPost` using a
systemd drop-in, reload systemd, then stop the runner. Its wrapper schedules
VM shutdown; cancel that shutdown if reconfiguring the run on the same VM.
Remove the temporary override and reload systemd before restarting the runner,
then enable the timer again. Resume only with the original frozen configuration
and checkpoint; a changed schedule uses a fresh run directory and W&B group.
