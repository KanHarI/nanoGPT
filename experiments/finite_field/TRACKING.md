# Weights & Biases tracking

Project: [kanhar/finite-field-curriculum](https://wandb.ai/kanhar/finite-field-curriculum).
Tracking is optional and disabled by default. Local JSONL logs and checkpoints
remain the source of truth; W&B provides shared curves and run comparisons.

Connection and upload were verified on 2026-10-16: 3 pilot runs, 15 main CPU
runs, and the stopped larger run. Remote loss/accuracy summaries were checked
against local results. All 12 unit tests passed; live online resume preserved
exact CPU weights, and an offline smoke test exercised final metrics and model
artifacts. Run URLs and verification details are recorded in
[wandb_verification.json](results/wandb_verification.json). The online smoke run
is isolated in the `tracking-smoke` group.

## Setup

```sh
uv pip install --python .venv/bin/python \
  -r experiments/finite_field/requirements-wandb.txt
.venv/bin/wandb login --verify
```

Run login in an interactive terminal and enter the API key privately. On a
headless GPU machine, provide `WANDB_API_KEY` through its secret/environment
configuration. Do not put credentials in source, configs, shell command
arguments, or chat. W&B 0.30.0 is the tested SDK version; the original CPU
dependency lock is retained separately to describe the historical environment.

The default project is `finite-field-curriculum`; the entity defaults to the
authenticated account. Override with `--wandb-project` and `--wandb-entity`.
Groups default to the output directory's parent name; choose an explicit group
for each matched experiment. Keep the same group across conditions and seeds.

## Train and resume

```sh
.venv/bin/python -m experiments.finite_field.train \
  --config experiments/finite_field/configs/pilot.json \
  --out-dir out/finite-field/tracked-pilot/curriculum-seed0 \
  --wandb online --wandb-entity kanhar --wandb-group tracked-pilot --final-eval

.venv/bin/python -m experiments.finite_field.train \
  --out-dir out/finite-field/tracked-pilot/curriculum-seed0 \
  --resume --wandb online --final-eval

.venv/bin/python -m experiments.finite_field.sweep \
  --config experiments/finite_field/configs/cpu.json \
  --seeds 0 1 2 3 4 --out-dir out/finite-field/tracked-cpu \
  --wandb online --wandb-entity kanhar --wandb-group tracked-cpu
```

`wandb_run.json` stores the remote ID, destination, group, and URL in each local
run directory. Online resume reuses that identity and requires the remote run to
exist. Entity and group are recovered from this file unless explicitly supplied;
pass the same `--wandb-project` again if using a nondefault project. Destination
or group changes are rejected on resume. Transfer this file together with the
checkpoint and local logs when moving a run. The experimental configuration must
also match; starting a new GPU experiment should use a fresh output directory.

W&B settings are separate from the scientific configuration, so enabling logging
does not invalidate existing checkpoints or alter matched-condition checks.
The SDK initializes before model seeding or RNG restoration. Exact CPU weight
equivalence is tested with tracking enabled and a resumed run, including dropout.

Train and evaluation records can share an optimizer update. We log them as
separate W&B events and use **`optimizer_step` as the chart x-axis**, allowing
both to be retained. W&B's internal `_step` is an event count, not an optimizer
update. After a crash, local resume trims metrics to the saved checkpoint;
already-uploaded W&B points cannot be rolled back by this code. Replayed updates
can therefore appear twice in remote history. Use committed local logs for the
final statistical comparison.

## Metrics and artifacts

The new GPU recipe uses `eval_mode="loss_grid"`. Its
[dedicated workspace](https://wandb.ai/kanhar/finite-field-curriculum?nw=7qw55xp483s)
has 96 plots: one train/test loss plot for each gamma 1–12 and field. Only the
192 loss series `eval/gamma_XX/field_YY/{train,test}_loss` and `optimizer_step`
are logged as history. Train/test are independent procedural banks, neither used
for optimization. W&B system statistics are disabled for this mode. Configuration
and completion metadata are still stored. See [GPU_RUN.md](GPU_RUN.md) for sample
counts and final-evaluation semantics. The metrics below apply to legacy CPU mode.

| Metric | Meaning |
| --- | --- |
| `train/loss`, `train/accuracy` | Answer-only metrics on the current training batch |
| `curriculum/gamma` | Actual gamma used at the logged update |
| `eval/high_gamma/loss`, `eval/high_gamma/accuracy` | Common fixed hard evaluation sample; compare conditions here |
| `eval/high_gamma/frequency_baseline_loss` | Per-field frequency predictor fitted on separate generated samples |
| `baseline/uniform_loss`, `baseline/uniform_accuracy` | Field-aware uniform reference: 2.01095 nats and 18.19% |
| `eval/high_gamma/by_field/*`, `by_operators/*`, `by_depth/*` | Evaluation breakdowns under the same prefix |
| `final/gamma_*/loss`, `final/gamma_*/accuracy` | Independent final evaluation at each requested gamma |
| `train/learning_rate`, `train/grad_norm` | Optimization diagnostics |
| `train/supervised_answers`, `train/characters` | Cumulative training exposure |
| `train/packed_rows`, `train/input_tokens` | Rows and padded input tokens in the logged batch |
| `train/mean_operators`, `train/mean_depth`, `train/rejection_rate` | Generator diagnostics |
| `throughput/answers_per_second` | Cumulative answers divided by training seconds, excluding evaluation/checkpoint time |
| `training/completed_updates`, `training/status` | Summary: complete, checkpointed, interrupted, or failed |

The full experimental config is attached to each run. Clean exit uploads a
small `experiment-report` artifact containing local `config.json`,
`environment.json` (including source hashes), and `final.json` when available.
`train --final-eval` and `sweep` send final metrics before closing the run.
Add `--wandb-checkpoint` to upload the saved model/optimizer checkpoint as a
model artifact at clean exit. Checkpoints are local by default. Their Python
state must only be loaded from trusted sources.

New runs record `include_division` explicitly in the config and have a
`division` or `no-division` tag, alongside the power toggle. Historical uploads
made before this option existed all used division; an absent flag on those
runs means division was enabled. They belong to a different experimental
distribution than the new division-disabled default.

## Import existing CPU results

```sh
.venv/bin/python -m experiments.finite_field.tracking \
  --import-run out/finite-field/cpu/curriculum-seed0 \
  --wandb online --wandb-entity kanhar --wandb-group cpu-baseline

.venv/bin/python -m experiments.finite_field.tracking \
  --import-run out/finite-field/cpu-large/curriculum-seed0 \
  --wandb online --wandb-entity kanhar --wandb-group cpu-large-stopped
```

Import reads history only through the latest checkpoint's committed record
count. It uploads final evaluation only if its step matches that checkpoint.
It does not train or evaluate the model. Imports have job type `history-import`;
their wall-clock upload time is not the original experiment time. Original
training seconds and optimizer updates are preserved. A duplicate import into
the same local run directory is rejected to avoid duplicate histories.

The 14,000-update CPU checkpoint is labelled `checkpointed` against its
20,000-update budget. It must not be included as a completed matched comparison.
The hard and mixture runs for that larger recipe do not exist.

## Offline mode

`--wandb offline` records SDK events locally without uploading. Later sync the
printed offline directory with `.venv/bin/wandb sync PATH`. This code rejects
offline resume into an existing identity because the SDK does not append to
offline history. Sync first, then resume with `--wandb online` to use the same
remote run. Alternatively, resume training with tracking disabled and use the
local metrics for analysis. Fully untracked runs can be imported afterward.

Local `wandb/` directories, outputs, environments, and `.env` files are ignored
by git. Source auto-upload and terminal-output capture are disabled; structured
metrics, configuration, and report artifacts are logged. SDK system metrics are
enabled only in legacy mode.
