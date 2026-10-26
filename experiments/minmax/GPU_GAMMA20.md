# Gamma-20 repeat with the same update budget

Cloud resources and VM checkpoints were deleted on 2026-10-18 after the
experiments finished. See [cloud cleanup](../CLOUD_CLEANUP.md). The VM paths and
operational instructions below are historical; W&B and saved reports remain.

This repeats the completed gamma-12 experiment with maximum gamma 20.
The first [curriculum run](https://wandb.ai/kanhar/minmax-curriculum/runs/09e0db5a)
launched on 2026-10-17 at approximately 13:08 UTC.
The user explicitly requested no additional optimizer updates. The recipe in
[gpu_gamma20.json](configs/gpu_gamma20.json) differs from the gamma-12
[recipe](configs/gpu_calibrated.json) only in `gamma_high` and `eval_gammas`.

| Setting | Value |
| --- | --- |
| Conditions | Curriculum, hard, exact-batch shuffled |
| Seeds | One: seed 0, fresh initialization for each condition |
| Total updates per condition | 26,668 |
| Ramp | Gamma 1 to 20, linear over 15,288 updates |
| Plateau | Gamma 20 for the remaining 11,380 updates |
| Model | 24 layers, width 256, 4 heads, FF width 1024; 18,964,736 parameters |
| Batch and context | 256 supervised answers; 256-token contexts |
| Answer budget per condition | 6,827,008 |
| Learning rate | 955 warmup updates, then constant 0.0006 |
| Precision | BF16 on the L4 GPU |
| Monitoring | 512 samples per split at every integer gamma 1–20, every 1,138 updates |
| Final evaluation | 4,096 fresh samples per split at every integer gamma 1–20 |

This preserves the prior update budget and ramp/plateau split. It does not add
updates or recalibrate a new schedule. Actual wall-clock time may increase
because gamma-20 expressions are larger and evaluation covers more gamma levels;
the previous 20-minute/20-minute timing is not an exact duration guarantee.

The AST cap remains 32 operators, and all other generator settings are identical:
uniform binary/ternary min/max and unary negation, signed atomic numerals from
-20 through 20, and no enforcement of alternation or removal of easy trees.
Whole newline-separated expressions are packed without splitting them. Only
answer tokens contribute to loss. Gamma is the parameter of a Poisson operator
count conditioned on being at most 32, rather than an exact operator count.

The shuffled condition replays the exact curriculum batches, including their
ASTs and answers, in a different order. Hard trains only at gamma 20. Both
controls receive the same update and answer budget as curriculum. Each new run
starts from scratch; the gamma-12 models are not used as pretrained weights.

## Separate outputs and monitoring

W&B project: `kanhar/minmax-curriculum`.
Group: `gpu-24layer-gamma20-matched-updates`. The new workspace has 20 panels
and 40 train/test loss curves: [gamma-20 dashboard](https://wandb.ai/kanhar/minmax-curriculum?nw=z0g0gef38xf).
The completed gamma-12 workspace and runs remain
available separately. Both train/test curves use independent procedural
evaluation samples, as in the previous experiment.

Persistent checkpoint paths on the VM:

```text
/home/kanhar/nanoGPT/out/minmax-gpu-24layer-gamma20-matched-updates/
  curriculum-seed0/checkpoint.pt
  hard-seed0/checkpoint.pt
  shuffled-seed0/checkpoint.pt
```

Each condition retains its config, source hashes, W&B identity, metrics, and
fresh final report. The sweep writes `summary.json` after all conditions finish.
The source snapshot from the completed gamma-12 run is archived before deploying
this repeat. The previous run directories and checkpoints are preserved.

The existing runner and 15-minute monitor read the same environment file,
`/home/kanhar/nanoGPT/minmax-run.env`, copied from
[gamma20.env](systemd/gamma20.env). This directs both training and completion
checks to the new directory. Two bounded recovery attempts, completion shutdown,
and the VM's 12-hour stop backstop remain in place. A stopped or Spot-evicted VM
cannot restart itself using the in-VM monitor.

The original `g2-standard-12` allocation could not restart because of zone
capacity. The retained `minmax-l4` VM in `us-central1-a` successfully restarted
as `g2-standard-8`, with the same NVIDIA L4 and persistent disk. Torch still
uses four CPU threads, and the experiment configuration is otherwise unchanged.

On 2026-10-17 at 13:51 UTC, Google Cloud evicted the Spot VM
(`compute.instances.preempted`). It remained stopped until the next status check;
the in-VM monitor could not recover while the VM was off. The retained VM was
restarted and the same W&B run resumed from checkpoint update 19,346. Fresh
optimizer progress was verified at update 19,665 at 16:02 UTC. Its latest
gamma-20 monitoring evaluation had loss 0.186724 and accuracy 94.73% on 512
test examples. The 26,668-update budget and all training settings were preserved.
See the [recovery record](results/gpu_gamma20_preemption.json).

The later [curve audit](GPU_GAMMA20_AUDIT.md) investigates the unexpected
shuffled loss plateau and jump. It verifies the actual batch schedules and
independently evaluates the checkpoints. Curriculum and hard-only have completed
and both reach approximately 95% accuracy at gamma 20; the audited shuffled
checkpoint was still training and showed large pre-clipping gradient spikes.
Shuffled subsequently completed all 26,668 updates with gamma-20 test loss
2.390637 and accuracy 32.96%. The [final comparison](results/gpu_gamma20_summary.json)
contains the matched three-condition endpoints.

```sh
# On the dedicated VM, from the repository root:
cp experiments/minmax/systemd/gamma20.env minmax-run.env
sudo install -m 644 experiments/minmax/systemd/minmax-*.service \
  experiments/minmax/systemd/minmax-watch.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl start minmax-runner.service
sudo systemctl enable --now minmax-watch.timer
```

For a manual run without systemd:

```sh
MINMAX_CONFIG=experiments/minmax/configs/gpu_gamma20.json \
MINMAX_OUT_DIR=out/minmax-gpu-24layer-gamma20-matched-updates \
MINMAX_WANDB_GROUP=gpu-24layer-gamma20-matched-updates \
bash experiments/minmax/gpu_run.sh
```
