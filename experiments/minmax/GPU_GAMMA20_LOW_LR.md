# Gamma-20 repeat at a lower constant learning rate

All three runs completed. On 2026-10-18 their final reports were preserved and
the cloud VMs, disks, and project were removed or shut down as documented in
[cloud cleanup](../CLOUD_CLEANUP.md). VM paths and launch instructions below
describe the original run environment.

The completed [gamma-20 experiment](GPU_GAMMA20.md) showed gradient instability
in shuffled; see the [audit](GPU_GAMMA20_AUDIT.md). This repeat halves the target
learning rate from 0.0006 to **0.0003**. The user requested that the shuffled
mixture run first, so the sequential order is **shuffled, curriculum, hard-only**.
Each condition starts from fresh seed-0 weights and a fresh optimizer.

The [new recipe](configs/gpu_gamma20_low_lr.json) differs from
[gpu_gamma20.json](configs/gpu_gamma20.json) only in `learning_rate`:

| Setting | Value |
| --- | --- |
| Model | 24 layers, width 256, 4 heads, FF width 1024; 18,964,736 parameters |
| Gamma range | 1 to 20; AST cap 32 operators |
| Curriculum schedule | Linear ramp over 15,288 updates, then 11,380 at gamma 20 |
| Updates and answers | 26,668 updates and 6,827,008 supervised answers per condition |
| Batch / context | 256 supervised answers / 256 tokens; whole expressions packed |
| Learning rate | Linear warmup for 955 updates, then constant 0.0003 |
| Other optimizer settings | AdamW, betas (0.9, 0.95), weight decay 0.1, clipping 1.0 |
| Seed / precision | Seed 0 / BF16 on the retained L4 VM |
| Monitoring | 512 samples per split at gamma 1–20, every 1,138 updates |
| Final evaluation | 4,096 fresh samples per split at gamma 1–20 |

The generator, batch seeds, permutation, evaluation banks, logging interval,
initialization, and update budget remain the same. Shuffled uses the exact
curriculum batches in a shuffled order; every batch has one gamma. It is the
same mixture as the previous run, not a new sampling scheme. Changing execution
order does not change any condition's seed or schedule. Old checkpoints are not
used to initialize this repeat.

## Outputs and operations

W&B project: `kanhar/minmax-curriculum`.
Group: `gpu-24layer-gamma20-lr3e-4`. The [lower-LR dashboard](https://wandb.ai/kanhar/minmax-curriculum?nw=1j8ea9b3j2q)
contains only the gamma 1–20 train/test loss panels, as before. Both curves use independent procedural
evaluation samples. Workspace and launch verification records are saved under
`results/gpu_gamma20_low_lr_*.json`.

The retained `minmax-l4` VM is in Google Cloud project
`finite-field-curriculum-kanhar`, zone `us-central1-a`, with an L4 GPU,
`g2-standard-8`, and the existing persistent disk. New checkpoints are written to:

```text
/home/kanhar/nanoGPT/out/minmax-gpu-24layer-gamma20-lr3e-4/
  shuffled-seed0/checkpoint.pt
  curriculum-seed0/checkpoint.pt
  hard-seed0/checkpoint.pt
```

Each arm saves its configuration, source hashes, metrics, W&B identity, and final
report. The completed comparison is written to `summary.json`. The earlier run
directories are preserved.

The runner and 15-minute watchdog both read `minmax-run.env`, installed from
[gamma20_low_lr.env](systemd/gamma20_low_lr.env). The wrapper's
`MINMAX_CONDITIONS` variable selects the execution order and defaults to the
previous order when unset. The monitor retains two bounded recovery attempts,
completion shutdown, and the VM's 12-hour stop backstop. As before, an in-VM
monitor cannot restart a VM that Google has Spot-evicted.

```sh
# On the dedicated VM, from /home/kanhar/nanoGPT:
cp experiments/minmax/systemd/gamma20_low_lr.env minmax-run.env
sudo systemctl daemon-reload
sudo systemctl start minmax-runner.service
sudo systemctl enable --now minmax-watch.timer
```

Equivalent manual execution:

```sh
MINMAX_CONFIG=experiments/minmax/configs/gpu_gamma20_low_lr.json \
MINMAX_OUT_DIR=out/minmax-gpu-24layer-gamma20-lr3e-4 \
MINMAX_WANDB_GROUP=gpu-24layer-gamma20-lr3e-4 \
MINMAX_CONDITIONS="shuffled curriculum hard" \
bash experiments/minmax/gpu_run.sh
```

This repeats the same number of optimizer updates; elapsed time depends on
throughput and is not a new 40-minute wall-clock limit. Results should be
compared against the previous run at matched update counts and interpreted as
a one-seed LR ablation.

The runner started on 2026-10-17 at 19:17:15 UTC on the retained L4 VM. The
configuration and all three batch plans were checked against the previous
recipe before launch. Shell syntax and the actual wrapper argument expansion
were checked for both its default order and the shuffled-first override. See
the [validation record](results/gpu_gamma20_low_lr_validation.json).

The first run is [shuffled-seed0 / 4c4da036](https://wandb.ai/kanhar/minmax-curriculum/runs/4c4da036).
At 19:18:17 UTC it had reached update 190; its step-zero evaluation grid matched
the previous shuffled run exactly. The live command confirmed the order
`shuffled curriculum hard`, a checkpoint existed, and both the trainer and
enabled watchdog timer were active. See the
[monitor verification](results/gpu_gamma20_low_lr_monitor_verification.json)
and [W&B launch verification](results/gpu_gamma20_low_lr_launch_verification.json).

## Completed results

All three conditions completed 26,668 updates. Their uploaded final reports,
configurations, source metadata, and W&B history were downloaded to
`out/minmax-gamma20-low-lr-final/` before cloud cleanup. The
[matched summary](results/gpu_gamma20_low_lr_summary.json) validates the original
configuration and the fresh final test bank of 4,096 examples at gamma 20.

| Condition | Gamma-20 test loss | Accuracy |
| --- | ---: | ---: |
| Shuffled | 0.172622 | 95.95% |
| Curriculum | 0.131232 | 96.41% |
| Hard-only | 0.122971 | 96.83% |

At LR 0.0006, shuffled had finished at loss 2.390637 and accuracy 32.96%.
Halving the LR substantially improved its final performance on this seed.
All three conditions learn at the lower LR; these endpoints do not establish
a final-performance advantage for curriculum over hard-only. The retained W&B
history supports comparisons of how quickly each condition learned.
Run IDs and final endpoints are listed in the
[final run record](results/gpu_gamma20_low_lr_final_runs.json).
