# Finite-field curriculum learning

Test whether gradually increasing expression complexity improves answer loss on
a fixed hard distribution compared with training on that distribution from the
start. Training examples are generated procedurally. The implementation reuses
the repository's `GPT` without changing the existing text-training workflow.

**CPU is the default. Division and exponentiation are disabled by default.** New
runs use `+`, `-`, and `*`. The scripts run
locally and do not create cloud resources. See [WORKPLAN.md](WORKPLAN.md) for the
experimental protocol and [RESULTS.md](RESULTS.md) for initial CPU observations.
See [TRACKING.md](TRACKING.md) for W&B setup, metric names, online resume, and
uploading existing results. The project is
[kanhar/finite-field-curriculum](https://wandb.ai/kanhar/finite-field-curriculum).
Google Cloud project details are recorded in [CLOUD.md](CLOUD.md).
The current GPU experiment uses 64 layers × 512 width on an H100, gamma 1–12,
one seed per condition, packed contexts, and constant LR after warmup.
See [GPU_RUN.md](GPU_RUN.md) for the measured batch/context choice and launch recipe.
The completed single-seed GPU comparison reached gamma-12 test loss **0.11312**
with curriculum and **0.45671** with shuffled exposure. Hard-only was stopped
early. See [GPU_RESULTS.md](GPU_RESULTS.md) for all final losses and limitations.

## Setup and quick start

Run commands from the repository root. Python 3.12 is a tested starting point.
`requirements-cpu-lock.txt` records the exact packages used for the initial
macOS/ARM CPU runs; use it instead of `requirements.txt` to reproduce that
environment. Platform-specific wheel availability can differ elsewhere.
All presets now disable division. Historical CPU results used division; add
`--include-division` to reproduce their operator distribution.

```sh
uv venv --python python3.12 .venv
uv pip install --python .venv/bin/python -r experiments/finite_field/requirements.txt

# Arithmetic, distribution, masking, causality, and exact-resume checks.
.venv/bin/python -m unittest discover -s experiments/finite_field/tests -v

# Deliberately memorize 16 items to verify the training path.
.venv/bin/python -m experiments.finite_field.sanity

# Short CPU pilot: all three conditions, one seed, 300 updates each.
.venv/bin/python -m experiments.finite_field.sweep \
  --config experiments/finite_field/configs/pilot.json \
  --seeds 0 --out-dir out/finite-field/no-division-pilot

# Larger CPU comparison: five paired seeds, 1,000 updates per run.
.venv/bin/python -m experiments.finite_field.sweep \
  --config experiments/finite_field/configs/cpu.json \
  --seeds 0 1 2 3 4 --out-dir out/finite-field/no-division-cpu

# Historical larger CPU recipe: 3.17M parameters, 20,000 planned updates per arm.
# Stopped at the user's request; use a new directory to reproduce it.
.venv/bin/python -m experiments.finite_field.sweep \
  --config experiments/finite_field/configs/cpu_large.json \
  --conditions curriculum hard mixture --seeds 0 --include-division \
  --out-dir out/finite-field/cpu-large-reproduction
```

The sweep runs conditions sequentially, avoiding competition for CPU threads.
Existing run directories are protected against accidental overwrites. To
continue an interrupted sweep, repeat its command with `--resume`. Completed
runs keep their final weights; their final evaluation is regenerated.

`configs/cpu_large.json` increases the model from 2 × 128 to 4 × 256
(3,173,632 total parameters versus 406,656), and the budget from 1,000 to 20,000
updates at batch size 32. This gives 640,000 supervised answers per condition.
The historical run retained division, gamma endpoints/ramp fraction, and the
disabled power toggle. Learning rate is 0.0003 with 200 warmup updates. Context remains
72, which fits every generated expression without spending CPU time on extra
padding. The first 100 CPU updates took 6.19 seconds, suggesting roughly 20–25
minutes of training per arm on this machine, plus evaluation and checkpointing.
This is a capacity-and-budget follow-up, not an isolated model-size ablation.
It was stopped during the curriculum arm, with a saved checkpoint at update
14,000; hard and mixture were not started. See [LARGER_CPU.md](LARGER_CPU.md).
The follow-up GPU recipe and W&B loss workspace are described in
[GPU_RUN.md](GPU_RUN.md). Historical CPU presets explicitly retain cosine LR;
new configurations and the GPU recipe use constant LR after warmup.

`configs/reference.json` preserves the larger proposed experiment: 4 layers,
256-dimensional embeddings, 64 answers per update, and 20,000 updates. It also
defaults to CPU; measure its cost before launching a sweep. A preset is a
starting recipe, not a claim that its budget is sufficient to learn the task.

## Arithmetic and serialization

The numeral alphabet is `0123456789abcdefghij`, with values 0 through 19.
Every logical number, including the field identifier, occupies one character.

| Prime order | Prefix | Permitted leaves/answers |
| --- | --- | --- |
| 2 | `2.` | `0`–`1` |
| 3 | `3.` | `0`–`2` |
| 5 | `5.` | `0`–`4` |
| 7 | `7.` | `0`–`6` |
| 11 | `b.` | `0`–`a` |
| 13 | `d.` | `0`–`c` |
| 17 | `h.` | `0`–`g` |
| 19 | `j.` | `0`–`i` |

Each item is `<field>.<postfix expression>=<answer>\n`, without spaces.
For example:

```text
7.21+4/=6
j.ii+=h
```

The first is `(2 + 1) / 4 = 6 (mod 7)`; the second is
`18 + 18 = 17 (mod 19)`. `j` identifies the largest field and is never a leaf
or answer in these fields. Operators are binary, with left/right order preserved
for subtraction, division, and exponentiation.

Every intermediate value is reduced modulo the selected prime. Optional division means
multiplication by the nonzero denominator's modular inverse. Any AST containing
division by zero is rejected, even if a later operation would discard its value.

Enable division with `--include-division` on `train` or `sweep`, or
`"include_division": true` in a JSON config. `--no-include-division` overrides
a config back to false. With division disabled, generation has no invalid
denominators and accepts every sampled AST on its first attempt. The example
`7.21+4/=6` illustrates the optional division-enabled format.

Enable exponentiation explicitly with `--include-power` on `train` or `sweep`,
or `"include_power": true` in a JSON config. `--no-include-power` overrides a
config back to false. Evaluation inherits the checkpoint's setting. When enabled,
the right subtree is evaluated in the field, and its canonical residue `0..p-1`
is used as an ordinary integer exponent. `0^0 = 1`. For example, `323-^` in F7
is `3^6`, because `2-3` evaluates to `6`. No negative or multi-character
integer exponent is introduced. The vocabulary always includes `^`, keeping
model shape constant when the toggle changes. Division and exponentiation can
be enabled independently; the vocabulary also always includes `/`.

## Procedural distribution and gamma

See [EXAMPLES.md](EXAMPLES.md) for ten generated and independently checked
expressions at each evaluation gamma (0, 1, 4, 8, and 16).

For each training item:

1. Choose a prime uniformly from the eight supported fields.
2. Draw operator count `K` from `Poisson(gamma)` conditioned on
   `0 <= K <= max_operators` (default 32). At gamma zero, `K = 0`.
3. Build a full binary AST. At each internal node, choose the left child's
   operator count uniformly from `0..K-1`; assign the remainder to the right.
   Choose an operator uniformly from those enabled (`+-*` by default) and each
   leaf uniformly from the field.
4. Reject invalid ASTs and redraw the entire tree **at the same field and K**.
5. Render the accepted AST as reverse Polish notation.

This split rule is not uniform over all binary tree shapes. Conditioning on
valid division also changes the accepted operator and shape distributions.
These choices are deliberate and identical in every arm. With division off,
there is no rejection. Training logs include
accepted operator counts, field counts, sampled sizes/depths, and rejection rate.
Resampling the field or K after a rejected tree would bias both distributions.

Gamma is the untruncated Poisson **rate**, not depth or an exact mean after
truncation. Each AST has `K+1` leaves and `2*K+1` expression characters. A full
line has `2*K+6` characters. The shifted model input needs `2*K+5` positions;
the configuration rejects contexts that could truncate an item. Gamma is never
provided to the model.

Training and evaluation share this generator, with independent random streams.
There is no finite training corpus or conventional train/validation split.
Repeats can occur naturally; no deduplication changes the sampling distribution.
Each monitored evaluation sample is procedurally generated once and replayed
across checkpoints. The final evaluation uses a separate seed and larger sample.

## Loss and batching

The tokenizer is a direct character lookup over
`0123456789abcdefghij.+-*/^=\n` (28 characters). With `pack_expressions=false`,
each batch row contains one complete item, right-padded with newline characters
to a fixed context length. The GPU recipe enables `pack_expressions=true`:
complete newline-separated items share a context whenever they fit, in sample
order. A row always begins at an expression boundary; no item is truncated or
split between rows. Ordinary causal attention can see earlier complete items in
the same row. The shifted final newline is excluded from the input.

All target positions are `-1` except the answer immediately after `=`. In shifted
next-token training, the logits **at `=`** predict that answer. The existing GPT
cross-entropy ignores `-1`, so there is exactly one supervised target per item.
The answer's later presence in the input and all padding lie in the causal
future and cannot leak into its prediction. Tests verify this property.

Loss is mean cross-entropy in nats over answers, using the full vocabulary
softmax. Accuracy uses full-vocabulary argmax; an operator, punctuation, or
out-of-field numeral prediction is wrong. We do not renormalize predictions to
valid field elements. No loss is assigned to newline generation.

`batch_size` counts expressions/answers, not packed context rows. Fixed batch size
matches supervised answers across arms. Packing makes the row count and dense
compute vary with sampled lengths, while context width remains fixed. Actual
characters, packed rows, generation work, and elapsed time are recorded locally.
Training time includes generation, batching, and optimizer updates; it excludes
evaluation, startup, and checkpoint writing.

## Conditions and evaluation

| Condition | Gamma at each update |
| --- | --- |
| `hard` | Always `gamma_high` |
| `curriculum` | Linear rise from `gamma_low` to `gamma_high` during the first `ramp_fraction` of updates, then constant |
| `mixture` | Exactly the curriculum's multiset of gamma values, shuffled across updates |

The base defaults are low=1, high=16, ramp fraction=0.4; the current GPU recipe
uses high=12. Updates are zero-indexed in
the scheduler and reported as completed-update counts in logs. The last ramp
update reaches high gamma. Every batch has one gamma in all three arms.
The mixture matches scheduled gamma exposure exactly, not the realized random
ASTs. Its schedule shuffle has its own seeded RNG.

Conditions with the same seed initialize the same model weights. Optimizer,
learning-rate schedule, number of updates, and answers per update are matched.
New configurations use linear warmup followed by **constant learning rate**
(`lr_schedule="constant"`). Historical CPU presets explicitly retain
`lr_schedule="cosine"` for reproduction. No restart occurs at
the curriculum transition. Field and operator choices use a separate training
RNG, so evaluation cannot consume training randomness.

The GPU recipe uses `eval_mode="loss_grid"`: independent procedural train/test
banks at every gamma 1–12 and every field. Neither bank is used by the optimizer;
both are fixed across checkpoints. Only these answer-loss curves are sent to
W&B. See [GPU_RUN.md](GPU_RUN.md) for sample counts, metric names, and the fresh
final evaluation. The following richer evaluation describes `eval_mode="legacy"`,
used by the CPU experiments.

In legacy mode, every monitored checkpoint is evaluated at the **same `gamma_high`**, using
exactly `eval_per_field` items from each prime. The primary endpoint is loss at
the final scheduled update on the independent final sample. No best-checkpoint
selection is used. Training loss follows different distributions across arms
and must not be used as evidence that curriculum wins.

Evaluation reports loss, accuracy, invalid-prediction rate, and breakdowns by
field, operator count, and AST depth. A label-frequency predictor is fitted to
a separate calibration stream for each field/gamma, with add-one smoothing,
then evaluated on the model's examples. Its majority-answer accuracy and loss
help detect success explained by label imbalance, including frequent zeroes.
Do not assume that random-expression answers are uniformly distributed.

Final evaluation also measures `eval_gammas` (always including `gamma_high`).
Results are equally weighted across fields. The comparison tool requires
matching configs, seed sets, and final evaluation sizes/seeds. It reports each
seed and paired loss differences: negative `curriculum_minus_hard` favors
curriculum. The 95% percentile bootstrap resamples **training seeds**, with no
interval for a single seed. Intervals from a few seeds remain exploratory;
evaluation items are not independent training replicates.

## Individual runs, resume, and artifacts

```sh
.venv/bin/python -m experiments.finite_field.train \
  --config experiments/finite_field/configs/cpu.json \
  --condition curriculum --seed 0 --out-dir out/finite-field/single

.venv/bin/python -m experiments.finite_field.train \
  --out-dir out/finite-field/single --resume

.venv/bin/python -m experiments.finite_field.evaluate \
  out/finite-field/single/checkpoint.pt \
  --output out/finite-field/single/final.json

.venv/bin/python -m experiments.finite_field.compare \
  out/finite-field/pilot/hard-seed0 \
  out/finite-field/pilot/curriculum-seed0 \
  out/finite-field/pilot/mixture-seed0 \
  --out-dir out/finite-field/pilot/comparison
```

Use `--stop-after N` to stop cleanly after N total updates without changing the
configured schedule horizon. Resume requires identical configuration, including
total steps and both operator toggles. Checkpoints store model, optimizer, training
sampler counters and RNG, Torch RNG, schedule position, and log position. Writes
replace the checkpoint atomically, and resume removes log rows beyond the saved
checkpoint. Exact CPU resume is tested even with nonzero dropout. Bitwise
reproducibility is not promised across hardware or PyTorch versions. Only load
trusted checkpoints: they include serialized Python state.

Saved runs predating the division toggle are interpreted as division-enabled
when resuming, evaluating, importing history, or comparing results. New partial
config files use today's division-disabled default. This distinction preserves
the old procedural distribution and prevents accidentally comparing it against
the new one. To resume an old run, use its saved config (`train --resume` without
`--config`), or pass `--include-division` when specifying a preset explicitly.
Saved runs without an `lr_schedule` field retain the historical cosine schedule;
new partial configs default to constant LR after warmup.

Each run writes:

- `config.json`: resolved parameters, including division and power toggles.
- `environment.json`: Python/PyTorch, platform, git revision, working-tree state,
  and SHA-256 hashes of Python sources and `model.py` (including uncommitted code).
- `metrics.jsonl`: sampled training metrics and fixed-target evaluation metrics.
- `checkpoint.pt`: latest resumable checkpoint, including the final update.
- `final.json`: independent final target evaluation and difficulty sweep.

The comparison directory contains `summary.json`, `learning_curves.png`,
`learning_curves.pdf`, and `difficulty_sweep.png`. Outputs under `out/` and the
local `.venv/` are ignored by git. The normal training entrypoint writes a
checkpoint; add `--final-eval` to also evaluate the independent final sample.
`sweep` runs final evaluation and plotting automatically. Add `--wandb online`
to either entrypoint to track runs; tracking is disabled by default.

## Code map

| File | Responsibility |
| --- | --- |
| `data.py` | ASTs, arithmetic, procedural generation, vocabulary, target masking |
| `config.py` | Validated settings, gamma schedules, learning-rate schedule |
| `train.py` | Training, metrics, checkpoint/resume, environment metadata |
| `tracking.py` | Optional W&B metrics, run identity/resume, historical import |
| `runtime.py` | Device configuration and optional BF16 autocast |
| `benchmark.py` | Actual-GPU batch/context measurements and ramp calibration |
| `workspace.py` | Focused W&B gamma/field train-test loss workspace |
| `evaluate.py` | Balanced procedural evaluation, baselines, checkpoint evaluation |
| `sweep.py` | Sequential matched runs and final reporting |
| `compare.py` | Config checks, paired statistics, exported plots |
| `sanity.py` | Tiny-sample overfitting check |
| `tests/test_experiment.py` | Correctness and reproducibility checks |
| `tests/test_tracking.py` | Tracking isolation, resume, and committed-history checks |
| `tests/test_gpu_pipeline.py` | Packing, loss-grid evaluation, LR, and batch selection |
