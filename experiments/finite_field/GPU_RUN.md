# H100 experiment: gamma 1–12

The VM and its checkpoint disk were deleted on 2026-10-18, and the Google Cloud
project was shut down, then subsequently restored. See [cloud cleanup](../CLOUD_CLEANUP.md). The operational
instructions and VM paths below are historical; saved reports and W&B remain.

Recipe frozen on 2026-10-16 after benchmarking the actual training VM.
Branch: `experiment/finite-field-curriculum`. Source hashes are saved with each
run; `source_snapshot.json` records the originating checkout because deployment
uses an archive rather than a Git clone.

[W&B loss workspace](https://wandb.ai/kanhar/finite-field-curriculum?nw=7qw55xp483s)
contains only this experiment's group, `gpu-64layer-gamma12`.
The first [curriculum run](https://wandb.ai/kanhar/finite-field-curriculum/runs/5c9ddae9)
started at 17:58 UTC on 2026-10-16 and completed all 48,210 updates. Its independent
final gamma-12 test loss was 0.11312. Hard-only was subsequently stopped early at
the user's request. Shuffled mixture also completed all 48,210 updates, with
final gamma-12 test loss 0.45671. See [GPU_RESULTS.md](GPU_RESULTS.md) for the
complete final comparison and archived reports. The VM stopped at 01:40 UTC
on 2026-10-17 and was confirmed `TERMINATED` in the post-run check; checkpoints
remain on its retained persistent disk.

## User-requested revision: stop hard-only

On 2026-10-16 at 22:40 UTC, hard-only was recorded as `early_stopped` after the
user requested proceeding to shuffled mixture. The retained checkpoint is update
17,688 of the planned 48,210; its monitored gamma-12 test loss is 1.81649. The last
observed optimizer log was update 18,180. The checkpoint, metrics, and original
configuration remain in `hard-seed0/`, with the decision in `stopped.json` and the
W&B run summary/artifact. The scientific budget was not shortened retroactively.

The active plan is now **curriculum and mixture**. Both the launcher and monitor
read `out/gpu-64layer-gamma12/run_plan.json`. Completed matching runs are reused
without reopening them, and hard-only cannot be restarted by this plan. Final
comparison and automatic shutdown require just the two planned full-budget runs.
Hard-only's partial curve remains visible in W&B but is excluded from final
paired comparisons. Its plateau over the observed budget does not establish
that it could never learn with more updates.

| Setting | Value |
| --- | --- |
| Model | 64 decoder layers, width 512, 8 heads, FF width 2048 |
| Parameters | 201,538,048, including position embeddings; tied input/output embeddings |
| GPU | One NVIDIA H100 80 GB, Google Cloud Spot `a3-highgpu-1g` |
| Arithmetic | Eight equally likely prime fields, `+ - *`; no division or powers |
| Precision | BF16 autocast, FP32 parameters/optimizer, fused AdamW, PyTorch SDPA |
| Context | 256 characters, complete newline-separated expressions packed together |
| Batch | 1,024 supervised expressions per optimizer update; no gradient accumulation |
| Updates | 48,210 per condition, 49,367,040 supervised answers |
| Conditions | Originally curriculum, hard-only, shuffled mixture; seed 0 each. Hard-only now early-stopped. |
| Learning rate | Linear warmup for 482 updates, then **constant 0.0003** |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1, gradient clip 1.0 |
| Gamma ramp | Linear 1→12 over the first 19,284 updates, then fixed at 12 |
| Evaluation | Every 536 updates, plus initialization and completion |

The exact recipe is [gpu_calibrated.json](configs/gpu_calibrated.json).
[gpu_64layer.json](configs/gpu_64layer.json) is the pre-calibration benchmark
input, not the launched update budget. No benchmark weights are reused.

## Batch/context measurements

[gpu_benchmark.json](results/gpu_benchmark.json) contains every measurement,
environment details, and the originally emitted benchmark config. The benchmark
ran before the explicit `lr_schedule` option was added: its timing loop already
used constant 0.0003. The launched config explicitly specifies `constant`.

Each measurement includes fresh procedural generation, packing, forward and
backward passes, gradient clipping, and an optimizer update. Candidates use
gamma 12, with three warmup updates followed by at least eight measured updates
and two seconds of measurement. A longer repeat verifies the selected result.

| Context | Batch expressions | Expressions/sec | Peak reserved GiB | Outcome |
| --- | ---: | ---: | ---: | --- |
| 72 | 512 | 3,731 | 24.99 | Fits |
| 72 | 1,024 | 3,414 | 65.29 | Fits, slower |
| 72 | 2,048 | — | — | Out of memory |
| 128 | 512 | 4,084 | 22.88 | Fits |
| 128 | 1,024 | 3,974 | 43.20 | Fits, slower |
| 128 | 2,048 | — | — | Out of memory |
| 256 | 512 | 4,213 | 22.08 | Fits |
| **256** | **1,024** | **4,569** | **55.54** | **Selected** |
| 256 | 2,048 | 4,815 | 75.81 | Less than 10% memory headroom |

Batches 32, 64, 128, and 256 were also measured at every context. The candidate
list extended to 4,096, but each context's search stopped at OOM or over 90%
reserved memory; 4,096 was never attempted. Selection uses the smallest batch
within 5% of the best measured throughput among candidates with 10% headroom.
This is the best tested configuration, not a claim of a global optimum.

The selected repeat measured 0.224 seconds/update, 4,569 expressions/sec,
39.46 GiB allocated and 55.54 GiB reserved. A batch of 1,024 unshared expression
rows at context 256 ran out of memory. With packing, gamma 12 averages about 128
rows per update and gamma 1 about 33 rows. Each row begins with a complete field
prefix, and no expression is split. Earlier complete expressions in a row remain
visible through ordinary causal attention; there is no attention reset at a
newline. Each expression's own answer remains causally hidden from its `=` logits.

## Schedule and fair comparison

For zero-indexed optimizer update `t`, curriculum uses
`gamma(t) = 1 + 11 * min(t / 19283, 1)`. Hard-only always uses 12. Mixture shuffles
exactly the curriculum's gamma multiset across updates. All arms share the model
initialization, update count, answers/update, LR, and evaluation streams.

LR is `0.0003 * (t + 1) / 482` during warmup and `0.0003` thereafter. There is no
decay or restart when gamma reaches 12. This is recorded in local and W&B configs.

Calibration at every integer gamma estimates a **50-minute training-time ramp**.
This is an update-based schedule calibrated from throughput, not a wall-clock
controller. Evaluation, checkpointing, and runtime variation add elapsed time.
Estimated training-only durations are about 2h38m each for curriculum and mixture
and 3h for hard-only; approximately 8h16m total, plus evaluation/checkpointing.
Packing makes easier batches cheaper, so dense compute and wall time are not
matched. Answer counts and optimizer updates are matched. A single seed cannot
establish robustness of any curriculum effect.

## Evaluation and W&B

The workspace has 12 gamma sections × 8 fields = 96 plots, each with train/test
loss. The logged keys are `eval/gamma_01/field_02/train_loss` and `test_loss`, and
the corresponding combinations through gamma 12 and field 19. `optimizer_step`
is the x-axis. Training diagnostics stay in local JSONL; W&B system statistics
are disabled for this recipe and automatic workspace panels are disabled.

As requested, train/test denote **independent procedural samples of the same
generator**, not seen versus unseen training examples. Neither bank is used for
optimization. Each monitoring bank contains 32 items per gamma/field, is fixed
across checkpoints, and is shared across conditions. Individual field curves
therefore have sampling noise. At completion, two fresh banks of 128 items per
gamma/field provide the final endpoint, logged at the final optimizer step.
Loss is answer-only cross-entropy in nats over the full character vocabulary.

## Operations

Live timing across updates 4,288–5,360 (two evaluation intervals, 148.03 seconds)
measured 120.54 seconds training (**81.43%**), 0.82 seconds evaluation (**0.55%**),
and 26.50 seconds checkpoint writing plus adjacent tracking (**17.90%**).
Remaining logging/loop overhead was 0.17 seconds (0.12%). Each evaluation sweep
took about 0.41 seconds; each 2.3 GB checkpoint save took about 13.25 seconds.
Training includes generation and packing. These are current early-ramp shares,
not whole-run averages; longer high-gamma updates will change the proportions.
The measurement observed normal log writes and atomic checkpoint replacements
at 5 ms resolution, without interrupting training or adding GPU work. Evaluation
and checkpoint estimates include their adjacent logging. Raw timing and method
are recorded in [gpu_timing.json](results/gpu_timing.json).

See [CLOUD.md](CLOUD.md) for SSH and VM details. The job runs under systemd, so it
survives terminal disconnects. Run from the repository root:

```sh
# Reproduce/resume on the prepared machine; completed arms reuse their checkpoint.
bash experiments/finite_field/gpu_run.sh

# Inspect the running service and recent output on the VM.
systemctl status finite-field-training
tail -n 20 training.log
```

The deployed service sets `AUTO_SHUTDOWN=1`: the wrapper schedules VM shutdown
two minutes after exit and records its exit code in
`out/gpu-64layer-gamma12/exit-code.txt`. The overnight monitor described below
intercepts that shutdown to verify completion or attempt bounded recovery.
A separate Compute Engine deadline stops
the VM 12 hours after its most recent start (`maxRunDuration` resets on restart).
Outputs live on the retained persistent
boot disk, not the disposable local SSD. Spot interruption can happen earlier;
resume from the last atomic checkpoint with the same script/config. Checkpoints
are saved at evaluation intervals; W&B points beyond the durable checkpoint may
be replayed after a crash. Stopping the VM ends GPU compute charges, while its
retained disk continues to incur storage charges.

### Overnight monitoring

At the user's request, `finite-field-watch.timer` checks progress **every 15
minutes**, independent of this chat or the local laptop. The sweep starts the
next active condition immediately when the preceding condition finishes;
the timer is an additional health check. A service exit also triggers an
immediate check, canceling the wrapper's pending shutdown while deciding whether
the sweep is complete or needs recovery.

`watchdog.py` records each check in `out/gpu-64layer-gamma12/watchdog.jsonl` and
the latest status in `watchdog.json`. It leaves active, progressing training
alone. If the trainer exits unexpectedly or has no progress for 30 minutes, it
resumes the original sweep/config from the existing checkpoints through
`finite-field-runner.service`. There are at most **two recovery attempts** across
the sweep. Persistent failure schedules shutdown while retaining logs and
checkpoints. Recovery follows the active run plan without changing the scientific recipe.

Completion requires matching final reports at the configured last update for
all active conditions, the saved comparison summary, and an exited training
service. The monitor then disables its timer and schedules shutdown in two
minutes. Disabling the timer permits later manual boot for artifact retrieval
without immediately shutting down again. The 12-hour Compute Engine runtime
limit remains a separate backstop. A Spot eviction stops this VM-side
monitor too; it cannot restart a stopped cloud instance.

On 2026-10-16 at 20:52:44 UTC (23:52 Israel time), Google preempted the VM during
the curriculum arm. The audit event is `compute.instances.preempted`, attributed
to `system@google.com`. The last uploaded evaluation was update 46,632, with
gamma-12 test loss 0.03549. At the next requested status check, the VM was
restarted at 21:14 UTC, and the monitor launched checkpoint recovery at 21:15 UTC.
This consumed one of its two allowed recovery attempts. Event details are in
[gpu_preemption.json](results/gpu_preemption.json).

```sh
systemctl list-timers finite-field-watch.timer
cat out/gpu-64layer-gamma12/watchdog.json
tail -n 5 watchdog.log

# Inspect a proposed action without executing recovery or shutdown:
.venv/bin/python -m experiments.finite_field.watchdog
```

The service/timer definitions and active trainer's exit hook are versioned under
`systemd/`. Six additional tests cover completion guards, progress checks,
bounded recovery, partial metric writes, and leaving healthy training untouched.

Verification before launch: 18 unit tests passed locally and on the VM, covering
arithmetic, masks/causality, packing, independent evaluation banks, constant LR,
tracking, and exact CPU resume. CUDA/BF16 training and finite gradients were
exercised by the full-size benchmark. The saved W&B workspace was read back and
verified to contain exactly 96 plots with the intended group filter.

Before committing the completed experiment, all 26 local tests passed, including
the monitor and revised run-plan checks. The 250-update CPU overfitting check
reached 100% accuracy and 0.00225 loss on its 16 fixed examples.
