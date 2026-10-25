# Proposed remote numerical-integral experiment

Proposal, 2026-10-25. The user requested a remote experiment using a large
network like the finite-field run. The preparation and execution code is now
implemented; see [REMOTE_RUN.md](REMOTE_RUN.md). The main budget remains subject
to GPU calibration. No cloud resources have been launched by this work.

## Model and hardware

Reuse the successful finite-field backbone, with fresh random initialization:

| Setting | Proposal |
| --- | --- |
| Decoder layers | 64 |
| Embedding width | 512 |
| Attention heads | 8, head dimension 64 |
| Feedforward width | 2048 |
| Vocabulary | 43 atomic integral tokens, tied input/output embeddings |
| Parameters | 201,545,728 with context 256; 201,480,192 with context 128 |
| Dropout / linear biases | 0 / disabled, as in the previous run |
| GPU | One H100 80 GB |
| Precision | BF16 autocast; FP32 parameters, gradients, and AdamW state |
| Attention | Existing PyTorch scaled-dot-product attention |
| Context | Start with 256; benchmark 128 against it before freezing |
| Initial batch candidate | 256 expressions per update; benchmark 64..1024 |

Parameter counts were checked with the current GPT implementation on PyTorch's
meta device, without allocating model weights or performing training. FP32
weights, gradients, and two Adam moments total approximately 3.00 GiB. Activations,
temporary buffers, CUDA workspaces, and allocator reservations are additional;
the batch size requires a measured GPU memory check. A model/optimizer checkpoint
is approximately 2.25 GiB before serialization overhead.

The previous [finite-field experiment](../finite_field/GPU_RUN.md) trained this
backbone on one H100 80 GB. That is direct evidence for the single-GPU choice,
but its batch size and examples/second do not transfer to longer integral records.

For continuity, the Google Cloud candidate is `a3-highgpu-1g`: one H100 80 GB,
26 vCPUs, 234 GB host RAM. Current Google documentation lists Spot or Flex-start
as the provisioning options for this size; availability, project billing, quota,
zone, and price must be checked when arranging a run. No live capacity or price
has been established by this design.
[Google Cloud GPU specifications](https://docs.cloud.google.com/compute/docs/gpus)

## The data preparation constraint

Generate labels on CPU **before the main GPU runs**, then train from immutable
shards on local storage. Examples still come from the procedural generator;
materialization changes when their labels are computed, not how they are sampled.
Do not silently substitute repeated epochs over the 20,000 survey examples.

Observed local figures:

| Measurement | Result | Limitation |
| --- | --- | --- |
| Quantized-label smoke | 192 accepted labels in 20.53 s, about 9.35/s | Small sample, one process, local CPU |
| Earlier unquantized survey | 20,000 values in 1,114.48 s, about 17.95/s | Six workers, different numerical/refinement policy |
| Previous finite-field GPU | About 4,569 examples/s at its selected batch | Different task and record lengths |

These are not forecasts of remote throughput. They demonstrate why inline
quadrature in the training loop would risk leaving the GPU waiting. The current
one-JSON-file-per-example cache is useful for smoke tests, but millions of files
and repeated rational validation must not become the training loader.

Proposed preparation sequence:

1. Measure accepted labels/s, rejection rate, and CPU scaling using the **actual
   quantized training labeler**, across gamma 1..20. Use separate worker processes
   because of mpmath contexts and signal deadlines. Benchmark worker counts before
   selecting a CPU machine; 64–128 vCPUs is an initial sizing candidate, not a
   throughput promise. Pin numerical libraries, policy, and timeouts.
2. Build independent pilot and evaluation samples. Audit labels using known
   analytic cases and a stratified sample checked with tighter or independent
   numerical integration, emphasizing discontinuities, near-zero outputs,
   rounding boundaries, and large magnitudes. Do not use the final test bank for
   hyperparameter selection.
3. Perform the brief GPU benchmark below using cached pilot data; discard its
   model and optimizer states. Freeze batch size, contexts, update counts, LR,
   seeds, evaluator sizes/cadence, and the data version.
4. Materialize the entire finite training plan: one corpus for curriculum and its
   exact shuffled replay, plus a separate gamma-20 hard-only corpus. Preserve
   within-batch order and all sample identities. Reject failed candidates at the
   original K and bounds, with no substitution of easier constraints.
5. Store compact token arrays plus offsets in shards, with checksums, batch
   manifests, raw reference values, quantized labels, uncertainties, and numerical
   provenance. Validate once on export/load; retain detailed rejection diagnostics
   separately. Freeze and verify the manifests before the main GPU allocation.

Required accepted training labels are approximately `2 * U * B`, where U is the
updates per condition and B the expressions per update: curriculum/shuffled share
one `U*B` corpus and hard-only needs another. Evaluation and pilot banks are extra.
For illustration, U=20,000 and B=256 would require about **10.24 million accepted
labels**, not 20,000; these are not the selected update/batch settings. Use measured
label throughput to estimate the CPU time and cost before committing to that size.
If it is too expensive, explicitly revise the update budget or propose a separate
finite-corpus experiment with reported reuse. Do not introduce reuse unnoticed.

## GPU calibration and budget

Start from the same 64-layer model for every candidate. Benchmark complete-record
packing at contexts 128 and 256 with 64, 128, 256, 512, and 1024 expressions/update,
stopping before unsafe memory pressure. Include gamma 1, 10, and 20, plus a stress
batch of long accepted records; the maximum record length remains 86 tokens.

Measure actual forward/backward/optimizer time, useful tokens/s, answers/s,
packing utilization, loader wait, and peak allocated/reserved VRAM. Select the
smallest batch within 5% of the best measured throughput while leaving at least
10% memory headroom. Verify it in a longer repeat. Avoid accumulation initially;
if it is necessary, normalize accumulated losses by the **total real-token count**
across microbatches, not an average of unequal microbatch means.

Proposed curriculum budget: **approximately 50 minutes of optimizer computation
ramping gamma, followed by 50 minutes at gamma 20**. This is a proposal based on
the earlier large-model experiment, not a claim that integration will converge
within that time.

Convert those times into fixed counts using the calibration:

- Choose ramp updates R so the predicted sum of update times over the linear
  gamma ramp is about 3,000 seconds.
- Choose plateau updates P so P times the measured gamma-20 update time is about
  3,000 seconds.
- Freeze U=R+P for **all conditions** and use ramp_fraction=R/U. The two phases
  need not contain the same number of updates.

For update index i starting at zero, the curriculum is:

```text
gamma(i) = 1 + 19 * min(i / (R - 1), 1)
```

Wall-clock time also includes evaluation, checkpointing, and loading. The
curriculum and exact shuffled plan should have similar total compute because
their packed batches are identical. Hard-only can take longer at the same U/B
because every example comes from gamma 20. Report updates, tokens, examples,
active training time, and elapsed time; do not describe all three as equal compute.
There is no fixed cloud-cost estimate until calibration, data volume, and current
resource prices are known.

Use AdamW, betas (0.9,0.95), weight decay 0.1, gradient clip 1.0. Start from the
previous successful LR **0.0003**, with linear warmup for about 1% of U, then
constant LR for the rest of training. Pilot instability can motivate a shared
lower LR before the comparison; do not tune individual conditions separately
or decay LR as gamma rises. All pilot weights are discarded.

## Conditions and interpretation

One paired model seed, initially seed 0. Run sequentially on one GPU:

| Order | Condition | Training exposure |
| --- | --- | --- |
| 1 | Curriculum | Linear gamma 1→20, then gamma 20 |
| 2 | Shuffled | A permutation of exactly the curriculum's cached batches |
| 3 | Hard-only | Fresh gamma-20 examples for every update |

Curriculum versus shuffled is the cleanest ordering comparison: identical
examples, labels, within-batch packing, and total token exposure. Hard-only tests
whether starting at maximum complexity works under the same update/example
budget, with a different data distribution and token count. All conditions start
from the same freshly initialized weights and use the same optimizer schedule.

Keep all 19 operators, the bounds grid including +/-2, domain restrictions,
K~Poisson(gamma) conditioned on 0..32, and the existing answer encoding. There is
no operator-introduction curriculum in this first study. Training minimizes
unweighted full-sequence NLL, with only padding masked.

Primary endpoint: final independent gamma-20 **answer NLL**, curriculum versus
shuffled. Secondary outcomes are the per-gamma answer curves, decoded numerical
accuracy, and the hard-only comparison. Falling full-sequence loss alone does not
demonstrate integration learning: formatting and predictable expression tokens
can improve without accurate answers. Fit answer-bit frequency baselines on a
separate calibration bank and report an always-zero numerical baseline. Also
report zero/nonzero targets, presence/absence of x, and trees with/without rounding
operators as diagnostic slices; x appearing syntactically does not prove that the
function is nonconstant. Preserve the unfiltered procedural distribution for the
primary results.

One seed is a first experiment, not a population estimate of training-seed effects.
Use paired per-example comparisons on the same evaluation banks; any reported
uncertainty from resampling those banks describes evaluation sampling only.

## Evaluation and W&B

Preserve gamma 1..20 and independent procedural `train`/`test` evaluation banks,
neither used in optimizer updates. Fixed monitoring samples stay shared across
conditions. Proposed starting sizes, finalized during calibration:

| Evaluation | Samples per gamma/split | Cadence |
| --- | ---: | --- |
| Teacher-forced NLL breakdown | 256 | Start, about every 5 calibrated training minutes, ramp end, completion |
| Unconstrained free answer generation | 32 from the monitoring bank | Start, about every 20 calibrated training minutes, ramp end, completion |
| Fresh final NLL bank | 2,048 | Once at the scheduled endpoint |
| Free generation on final bank | 256 | Once at the scheduled endpoint |

Freeze optimizer-step evaluation intervals after timing them. Aim for no more
than about 10% evaluation overhead. If decoding is too costly, reduce its frequency
first while retaining all gamma levels. The implemented evaluator separates loss
and decoding cadences and batches compatible prompt lengths across the whole grid,
keeping their result identities. A KV cache is a possible later
optimization if measurements justify it, not a prerequisite assumed to exist.

Log all implemented losses: answer, full, non-answer, sign, exponent, mantissa,
and answer-token fraction. Log free-generation exact quantized accuracy, valid
answer rate, numerical tolerance accuracy, and MAE among valid answers (absent
when none are valid). Preserve the current atol=1e-6/rtol=0.01 tolerance and
unconstrained decoding; no true answer prefix is supplied. Use optimizer updates
as the main x-axis, with training time and token counts also available.

The proposed W&B project is `numerical-integrals-curriculum`, with one group for
this frozen recipe. Upload configuration, data manifest/checksums, numerical-policy
version, source revision/hashes, environment, and final report. Display answer NLL
first, with other losses and numerical metrics in separate panels.

## Persistence and readiness

Save atomic checkpoints about every five calibrated training minutes and at
phase boundaries. Keep the latest two and the final checkpoint for each condition,
plus the immutable data manifests. Checkpoints must include model, optimizer, RNG,
step, shard/batch position, and data version; sync to storage that survives VM
deletion. Confirm a remote interruption/resume test before a Spot run. Final
artifacts and W&B syncing must finish before automatic GPU shutdown.

The package implements the model, full-token objective, numeric encoding,
curriculum/hard/shuffled plans, metrics, checkpoints, and W&B logging, together with:

- Parallel resumable labeling/export, immutable training shards and batch manifests,
  and efficient prefetching from those shards.
- The integral-specific GPU batch/context benchmark and schedule calibration.
- Independent loss/decoding cadence and better batching across evaluation banks.
- Remote checkpoint persistence, resume validation, sequential-run orchestration,
  and automatic shutdown after artifact verification.

CUDA training requires a complete prepared corpus and never falls back to live
quadrature. CPU tests cover export/recovery, data identity, matched comparisons,
and archival integrity. The actual H100 benchmark, remote CUDA resume preflight,
durable storage setup, and measured data-volume/compute budget remain launch-time
work. The runbook distinguishes the provisional benchmark config from the frozen
recipe produced by calibration.
