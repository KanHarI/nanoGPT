# Signed min/max curriculum learning

An implemented procedural RPN task with binary and ternary min/max, unary
negation, and one character for every integer from -20 through 20. The generator
does not enforce alternation, simplify expressions, reject ties, or remove easy
trees. See the [design document](../proposals/NESTED_MIN_MAX.md) and
[comparison protocol](../proposals/README.md).

The default model is **24 layers, width 256, 4 heads, FF width 1024**, with
**18,931,968 parameters** including position embeddings. This is about 10.6 times
smaller than the 201.5M-parameter finite-field model. Each head has width 64.
Size is configurable; this is a starting capacity, not a claim of convergence.
The earlier 4-layer, width-128 CPU results retain their original recipe in
[cpu_4layer.json](configs/cpu_4layer.json). The 24-layer model has been checked
with a CPU forward/backward pass. See [GPU run operations](GPU_RUN.md) for the
larger experiment and its calibrated recipe.
The [gamma-20 lower-LR repeat](GPU_GAMMA20_LOW_LR.md) runs shuffled first at a
constant post-warmup LR of 0.0003, followed by the two matched controls.
All three runs completed. The [cloud cleanup](../CLOUD_CLEANUP.md) on 2026-10-18
deleted the VMs and checkpoint disks and shut down the project; final reports
and W&B results remain available.

## Quick start

Run from the repository root, using the existing `.venv`. Dependencies are shared
with the finite-field experiment; no additional packages are required for local
training. For a new environment:

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r experiments/finite_field/requirements.txt
```

```sh
# Exact arithmetic, sampling, masking, curriculum/replay, and resume checks.
.venv/bin/python -m unittest discover -s experiments/minmax/tests -v

# Verify that the model can memorize a fixed 16-example sample.
.venv/bin/python -m experiments.minmax.sanity

# Print ten independently generated training lines at gamma 12.
.venv/bin/python -m experiments.minmax.data --gamma 12 --count 10 --seed 0

# Short CPU smoke comparison: 300 updates, 2 layers x 64, all three conditions.
.venv/bin/python -m experiments.minmax.sweep \
  --config experiments/minmax/configs/pilot.json \
  --out-dir out/minmax/my-pilot --seeds 0

# Current 18.93M-parameter recipe, CPU by default, 2,000 updates per condition.
.venv/bin/python -m experiments.minmax.sweep \
  --config experiments/minmax/configs/small.json \
  --out-dir out/minmax/my-24layer-comparison --seeds 0

# Train one condition; optionally stop at a checkpoint without changing its schedule.
.venv/bin/python -m experiments.minmax.train \
  --config experiments/minmax/configs/small.json \
  --out-dir out/minmax/my-curriculum --stop-after 400
.venv/bin/python -m experiments.minmax.train \
  --out-dir out/minmax/my-curriculum --resume

# Re-evaluate a trusted saved model, with an explicitly separate report.
.venv/bin/python -m experiments.minmax.evaluate \
  out/minmax/my-curriculum/checkpoint.pt --device cpu \
  --gammas 1 4 8 12 16 --examples 1024 --output out/minmax/reevaluation.json
```

Existing nonempty output directories are protected. A sweep with `--resume`
continues interrupted arms and reuses completed arms only when their saved final
configuration matches. Re-evaluations are marked separately and cannot be passed
off as the original matched final endpoint.
Use a fresh output directory for the new architecture. Existing checkpoints
resume with their original saved configuration.

## Tokens and arithmetic

| Tokens | Meaning |
| --- | --- |
| `0`–`9` | Integers 0–9 |
| `a`–`k` | Integers 10–20 |
| `A`–`T` | Integers -1 through -20, respectively |
| `m`, `n` | Binary minimum, maximum |
| `t`, `u` | Ternary minimum, maximum |
| `~` | Unary negation |
| `=`, newline | Answer separator and item separator |

There are 48 characters, including 41 numeral tokens. Case matters: `T` is -20,
while `t` is ternary minimum. `C` is the literal -3; `3~` is an expression that
computes -3. Comparison uses decoded signed integers. All intermediates and
answers remain in [-20, 20].

```text
120u312um=2
C2m=C
T~=k
3~2n=2
```

The first line is `min(max(1, 2, 0), max(3, 1, 2)) = 2`, preserving both ternary
max nodes. There is no field prefix. Loss supervises only the answer after `=`;
the logits at `=` predict that answer. Newline, expression, and padding targets
are ignored. Predictions use the full vocabulary softmax; predicting an operator
or separator is incorrect. No valid-numeral renormalization is applied.

A gamma-12 sample from generator seed 10 is:

```text
dMKCRignmnufHSuD2tcjNttkeFft22unt=N
```

It evaluates to -14 (`N`) and has 12 operators, AST depth 6, and maximum RPN
stack occupancy 7. Other gamma-12 samples can have different operator counts.

## Distribution, batching, and schedules

Draw operator count K from Poisson(gamma), conditioned on 0 <= K <= 32. At each
internal node choose uniformly among the five operators, independent of its
ancestors. Split the remaining operator budget uniformly among ordered weak
compositions for that node's arity. Leaves are uniform over all 41 integers.
`operator_weights` can explicitly configure a different fixed distribution;
the default is `[1, 1, 1, 1, 1]` in the order `m, n, t, u, ~`.

Every input is kept as sampled, including same-operator chains and double
negations. The recursive AST evaluator supplies labels; a separate stack oracle
checks the arithmetic in tests. Tree depth, RPN stack occupancy, and alternation
are different properties and are reported separately.

Each complete line has at most `3*K + 4` characters (100 at K = 32). The default
128-character context fits every line. Whole newline-separated lines are packed
in sample order; a context starts at an expression boundary. Ordinary causal
attention can see previous complete examples in the row. No expression is split.
Batch size counts supervised answers, not packed rows.

The main recipe has 2,000 updates, 64 answers/update, a linear gamma 1-to-12
ramp over 800 updates, and a plateau afterward. AdamW uses 40 warmup updates then
**constant LR 0.0006**, betas (0.9, 0.95), weight decay 0.1, and clipping at 1.0.
The pilot has 300 updates and 32 answers/update on a 109,888-parameter model.
These are CPU development budgets. The GPU benchmark in
[benchmark.py](benchmark.py) chooses a batch/context pair on the actual device
and measures timing across gamma. The current GPU run uses these timings for
a 20-minute gamma ramp plus a 20-minute gamma-12 plateau; see [GPU_RUN.md](GPU_RUN.md).

| Condition | Training order |
| --- | --- |
| `curriculum` | Linear gamma ramp, then high-gamma plateau |
| `hard` | High gamma at every update |
| `shuffled` | Exactly the curriculum batches, permuted across the whole run |
| `shuffled_ramp` | Permute just the ramp; preserve the identical plateau batches and order |

The default sweep runs curriculum, hard, and shuffled. Add the optional fourth
arm with `--conditions curriculum hard shuffled shuffled_ramp`. Conditions share
initial weights for each seed. Every logical batch has its own deterministic
seed and gamma; its ASTs can be regenerated at any step. Shuffling preserves the
actual examples, target counts, and character totals, not just expected exposure.
There is no hidden stateful data stream whose trajectory changes after resume.

The implementation reuses nanoGPT's `GPT`, the finite-field experiment's general
truncated-Poisson weight calculation, and its device/autocast helpers. It does not
modify the finite-field training path. CPU defaults use float32; the GPU recipe
uses CUDA and BF16. Exact interrupted/uninterrupted equality is tested on CPU;
GPU kernels can introduce numerical differences after a restart.

## Evaluation and W&B

At every gamma 1–12, independent procedural `train` and `test` banks monitor the
model; neither is optimizer data. Banks remain fixed across checkpoints and
conditions, and the final report uses fresh seeds and larger samples. The main
recipe uses 128 items per bank/gamma for monitoring and 512 for the final report;
the pilot uses 32 and 128. Natural duplicates are possible, especially at low gamma.

Reports include loss, accuracy, invalid predictions, and slices by root operator,
operator count, depth, stack occupancy, negations, ternary count, alternations,
and answer value. Independent calibration samples fit label-frequency and
root-conditioned label-frequency baselines. Global-extreme heuristics and their
coverage are reported separately. Uniform numeral guessing has loss ln(41).
Two fixed structure-sensitive pairs provide a small diagnostic, not a benchmark
large enough to establish generalization by itself.

W&B is optional and disabled by default. Enable it on `train` or `sweep`:

```sh
uv pip install --python .venv/bin/python -r experiments/finite_field/requirements-wandb.txt
.venv/bin/python -m experiments.minmax.sweep \
  --config experiments/minmax/configs/small.json \
  --out-dir out/minmax/tracked-comparison \
  --wandb online --wandb-entity kanhar --wandb-group minmax-small-seed0
```

The default project is `minmax-curriculum`. Logged chart metrics are only
`eval/gamma_01/train_loss`, `eval/gamma_01/test_loss`, and the corresponding pairs
through gamma 12, indexed by `optimizer_step`. Richer diagnostics stay in local
JSON. Clean exit uploads small report artifacts, not model weights. `--wandb
offline` supports local logging; online resume reuses the saved run identity.
Offline W&B history cannot be resumed directly; sync it first and resume online,
or continue with tracking disabled. No cloud machine is provisioned by these tools.

## Outputs and initial validation

Each arm writes `config.json`, `environment.json` (source hashes and model size),
`metrics.jsonl`, an atomic `checkpoint.pt`, and a fresh `final.json` at completion.
Checkpoints contain model/optimizer state, RNG state, and the next update index.
Resuming requires the unchanged recipe and truncates log entries newer than the
last checkpoint. A forced stop retains the most recent saved checkpoint; there
is no signal-triggered final checkpoint. Sweeps write `summary.json` and reject
partial runs, incompatible settings, and unmatched seed sets when comparing.

See [RESULTS.md](RESULTS.md) for CPU validation observations and their limits.
The gamma-20 repeat keeps the completed gamma-12 run's exact update budget;
see [GPU_GAMMA20.md](GPU_GAMMA20.md) for its recipe and separate output paths.
The 24-layer GPU comparison is described in [GPU_RUN.md](GPU_RUN.md), with
measured batch/context selection, a calibrated ramp, online W&B, and automatic
monitoring. Larger structural stress suites, true size extrapolation, and a
full-budget capacity comparison remain follow-up experiments. Symbolic
differentiation remains a proposal and has not been implemented.
