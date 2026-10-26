# Twenty-minute shuffled gamma 1–5 run

Authorized on 2026-10-25 in Google Cloud project `personal-383909`, following
the [ten-minute gamma 1–5 diagnostic](GPU_GAMMA5.md). This is a fresh training
run from seed-0 initialization, not a continuation of its checkpoint.

The model is unchanged: 64 layers, embedding width 512, 8 heads, feedforward
width 2048, 201,480,192 parameters. Training uses one H100 80 GB, BF16 autocast,
16 expressions per update, and context 128 with whole-expression packing and
newline boundaries. Full-sequence next-token NLL remains the objective.

The budget doubles to 20,608 updates, approximately 20 minutes of training and
evaluation based on the previous measurement. CPU preparation, VM startup,
checkpoint saves, and transfers are additional. The underlying gamma ramp
doubles to 10,424 updates from 1 to 5, followed by a plateau at 5; the entire
batch schedule is shuffled before training. Thus this is the same mixture
distribution as before, rather than a uniform choice of integer gamma values.
Gamma is always in [1, 5], while actual AST operator count is a Poisson draw
with the existing cap of 32.

LR warms up over 103 updates to 0.0003, then remains constant. All other model,
optimizer, evaluation, and generator settings remain unchanged. Loss banks are
evaluated every 1,030 updates and at required boundaries. Free decoding and
checkpoints run every 5,150 updates and at boundaries. Evaluation covers gamma
1, 2, 3, 4, 5 only, using independent procedural train/test banks: 128 examples
per monitoring cell and 256 per final cell, with 8 and 16 free-decoding prompts,
respectively. The evaluation seed is kept fixed for comparison with the short
run; these banks never become optimizer training data.

Generate 329,728 training samples and 6,400 evaluation/baseline samples, each
training batch consumed once. No fixed training pool is replayed. Numerical
labels are prepared with 48 CPU workers before starting the H100. Failed
numerical evaluations are rejected by the existing labeler.

The previous gamma 1–5 run measured 616.7 accepted labels/s on those 48 CPU
workers versus 291.5 training examples/s on the H100. Therefore preparation is
not an average-throughput bottleneck with the separate CPU VM at these gammas.
The current implementation serializes preparation and training, adding startup
latency. A buffered producer could overlap them; that streaming pipeline has
not been implemented or measured. These rates do not establish that labeling
on the H100 VM's fewer CPU cores would keep up.

Operational files and frozen configuration:
`out/integrals-h100-shuffled-gamma5-20m-20261025/`.
The reproducible [run configuration](configs/h100_shuffled_gamma5_20m.json)
is also retained alongside the experiment code.
Remote working directory: `/home/kanhar/nanoGPT-gamma5-20m` on the existing
`integrals-data-smoke` and `integrals-h100-smoke` VMs in `us-central1-a`.
The controller verifies data/checkpoint checksums, relays W&B metrics locally
without copying credentials, downloads the checkpoint, and stops both VMs.
The CPU VM has a two-hour safety cutoff; the GPU has a 45-minute cutoff and
its training service a 35-minute timeout. Boot disks are retained.

The [archived dashboard](https://wandb.ai/kanhar/numerical-integrals-curriculum?nw=oqppf6lmos9)
is filtered to this run. It keeps five collapsed gamma sections, paired
train/test curves, logarithmic Y axes, and a minimum displayed optimizer update
of 1,000. Full histories remain logged. The earlier ten-minute run has a
[separate archived view](https://wandb.ai/kanhar/numerical-integrals-curriculum?nw=c0oyk84007t).

Data preparation completed in 561.02 seconds at 599.14 accepted labels/s with
48 workers. It generated all 336,128 labels and rejected 4,906 failed attempts.
The complete corpus was verified before stopping the CPU VM and starting the
H100. Its manifest SHA-256 is
`769041da43620421ca39a12c81cbfb2334786fda242e809139fe851fc161dbbb`.

Training launched at approximately 14:13 UTC on 2026-10-25.
[W&B run `57m5nihk`](https://wandb.ai/kanhar/numerical-integrals-curriculum/runs/57m5nihk)
is named `shuffled-gamma1-5-fresh-20min-seed0`. All 20,608 updates completed;
results and the verified checkpoint were downloaded before stopping the H100.
Both VMs were independently confirmed `TERMINATED`. Their boot disks remain.

## Results

The [result JSON](results/h100_shuffled_gamma5_20m.json) records the full evaluation
grids and configuration. Fresh final test answer NLL is **0.37993 nats/bit**
(0.42681 for the ten-minute run on the same banks). Full-sequence NLL is 0.89732.
The independent bit-prior baseline is 0.60099 nats/bit.

Greedy decoding produced 80 valid answers, of which **14/80 (17.5%)** were exact
and within tolerance, versus 2/80 after ten minutes. Always predicting zero gets
10/80 (12.5%) on the same prompts. This is improvement, with a small decoding
sample and substantial remaining numerical error; it is not a curriculum result.

Tracked time was 1,080.24 s training, 45.02 s evaluation, and 14.80 s loading:
1,140.06 s total, excluding initialization, checkpoint saves, and transfers.
The full checkpoint is at
`out/integrals-h100-shuffled-gamma5-20m-20261025/results/out/shuffled20m/shuffled-seed0/checkpoint.pt`
and on the retained boot disk under the remote working directory.

This run used the original fixed-gamma-per-batch schedule. The later
`training_gamma_sampling="uniform_integer"` change applies to new runs and did
not alter this completed experiment.
