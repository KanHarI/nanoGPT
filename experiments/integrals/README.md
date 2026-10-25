# Numerical-integral curriculum

Full-sequence next-token NLL training on domain-checked numerical-integral ASTs.
The implementation reuses the [documented generator and numerical checks](../proposals/NUMERICAL_INTEGRALS.md),
adds stable quantized targets, and trains the existing causal GPT. No cloud
deployment is involved. The CPU smoke configuration is a pipeline check, not a
learning experiment or a choice of the eventual GPU model/budget.

The [proposed remote experiment](REMOTE_EXPERIMENT.md) uses the 64-layer,
width-512 finite-field backbone and lays out CPU label preparation, H100
calibration, matched controls, and evaluation. The [remote runbook](REMOTE_RUN.md)
gives the implemented export, calibration, training, resume, durable-archive, and
optional shutdown commands.

## Records and vocabulary

There are **43 atomic tokens**. Spaces below are only for readability. A record is:

```text
lower upper <integrand RPN> INT_X = S? <EXP> E? E? E? E? E? <MANT> M? M? M? M? M? M? M? M? \n
```

For example, the integral of x from 0 to 1 is exactly 0.5:

```text
0 1 x INT_X = S0 <EXP> E1 E0 E0 E0 E0 <MANT> M1 M0 M0 M0 M0 M0 M0 M0
```

There is an actual newline token after each complete record. Each context starts
with a newline, so the first lower-bound token is also a next-token target.
Multiple complete records are packed into one context; no expression is split
or truncated. Causal attention can see earlier complete records in the same row.
`<PAD>` fills unused positions and its targets are ignored. With at most 32
operator nodes, the longest record is 86 tokens; the default context is 128.
`batch_size` counts examples per update, not packed rows.

Bounds are two distinct uniformly sampled points from
`-2, -1, NEG_HALF, 0, HALF, 1, 2`, sorted: all 21 intervals have equal probability.
Integrand leaves are x and `-1, 0, 1, PI, E, 2, 3, HALF, THIRD`.
The 19 operators are:

- Binary: `+`, `-`, `*`, `/`.
- Unary: `NEG`, `SQUARE`, `SQRT`, `SIN`, `COS`, `TAN`, `ASIN`, `ACOS`, `ATAN`,
  `LN`, `EXP`, `W0`, `FLOOR`, `CEIL`, `FRAC`.

Domain restrictions, conservative ranges, and the fixed per-node magnitude guard
of 1e6 from the preview remain active. FRAC means u-floor(u), including for negative
u. Operator count K is Poisson(gamma), conditioned on 0..32. It is not fixed to
gamma. Unary/binary choices have equal probability; operator choice is uniform
among eligible operators of the selected arity. There is no forced alternation.

## Numeric labels

Each answer has 1 sign bit, 5 exponent bits, and 8 **explicit** mantissa bits,
MSB first, plus two separator tokens:

```text
value = (-1)^S * (M / 256) * 2^(E - 16)
```

Normals have M=128..255. At E=0, M=1..127 are subnormals. Zero has only one
encoding: S=0, E=0, M=0. Range is -32640..32640, with smallest positive value
2^-24. Exact rational arithmetic implements round-to-nearest, ties-to-even,
including normalization carries and subnormals. Values below half the smallest
subnormal round to canonical zero; exact half also rounds to zero. Values beyond
the supported range are rejected, never clipped. There are 8,447 canonical values.

For each candidate, the numerical evaluator uses two agreeing quadrature checks
and estimated errors, with high-precision fallback. The labeler constructs an
uncertainty radius equal to four times the largest of:

- The estimated errors of the last two successful checks.
- Their deviations from the selected estimate.
- A numerical floor: max(1e-12, 32 float ULPs), or 1e-30 after two high-precision checks.

Both ends and the center must encode to the same answer. A rounding-boundary or
range failure first gets a high-precision retry on the same AST. Persistently
unstable, out-of-range, ambiguous-discontinuity, or timed-out candidates are
discarded and replaced **at the same K and bounds**, up to 100 attempts. There is
no fallback to easier constraints. This is a conservative engineering acceptance
rule based on numerical error estimates, not a rigorous enclosure proof; narrow
features can still challenge numerical quadrature, especially with FLOOR/CEIL/FRAC.

Accepted examples, checked values, uncertainties, numerical diagnostics, and rejected
candidate RPN/reasons are cached under `cache_dir`. Cache hits revalidate canonical
encoding and label stability. Seeds depend on the data version and stream identity.
Use the same cache for comparisons: wall-clock timeouts can otherwise change which
candidate is accepted on different machines. Do not edit the label policy without
bumping `data.VERSION` or selecting a fresh cache. The local labeler uses Unix
signal deadlines in the main thread. Remote preparation runs independent worker
processes and exports immutable, checksummed shards. CUDA training requires those
shards and never performs live quadrature. See the remote runbook for throughput
measurement and corpus sizing.

## Objective and W&B metrics

The optimizer minimizes **ordinary unweighted NLL over every real next token**:
bounds, integrand, INT_X, '=', numeric bits, separators, and newline. Only padding
is masked. The softmax always covers all 43 tokens; there is no token-family or
grammar masking. Longer expressions therefore contribute more non-answer targets.

The main task metric is answer NLL. For every gamma 1..20 and each of `train` and
`test`, W&B gets the following, against `optimizer_step`:

| Metric under `eval/gamma_XX/{train,test}/` | Definition |
| --- | --- |
| `answer_nll` | Mean NLL over the 14 numeric bits, nats/bit; excludes separators |
| `full_nll` | Mean NLL over all real target tokens, nats/token; training objective |
| `non_answer_nll` | Mean NLL over all remaining targets, including separators and newline |
| `sign_nll`, `exponent_nll`, `mantissa_nll` | Mean NLL over 1, 5, and 8 respective bits |
| `answer_token_fraction` | Numeric bits divided by all real targets |
| `exact_answer_accuracy` | Greedy free-generation matches the complete canonical answer |
| `valid_answer_rate` | Greedy answer has the correct layout and a canonical number |
| `decoded_mae` | Mean absolute error against the **unrounded integral**, among valid predictions |
| `tolerance_accuracy` | Fraction with error <= absolute_tolerance + relative_tolerance * abs(reference) |

Counts (`examples`, `tokens`, `answer_bits`, `generation_examples`, `valid_answers`)
are logged too. Losses aggregate token sums/counts, never averages of unequal batch
means. Full/answer/non-answer and sign/exponent/mantissa therefore decompose exactly.

NLL is teacher-forced. The separate generation metrics start from just the bounds,
AST, and '=' and freely predict all 16 answer tokens, including both separators.
They never receive gold preceding answer bits. Greedy decoding is unconstrained;
invalid answers count as wrong for both accuracies. MAE is absent when no answer
is valid; always read it alongside validity. Tolerance defaults to atol=1e-6 and
rtol=0.01. No unstable division by a near-zero reference is needed.

As previously agreed, `train` and `test` mean **independent procedural evaluation
banks**, neither used for optimizer updates. Monitoring banks stay fixed across
checkpoints and conditions. Final results use fresh independent banks and a
separate `final/gamma_XX/...` namespace. Natural repetition of simple ASTs is
possible; independent sampling does not guarantee novel expressions.

Actual optimizer-batch metrics use `train/full_nll`, `train/answer_nll`, etc., plus
gamma, learning rate, and gradient norm. Labeling, optimizer computation, and
evaluation times are recorded separately. `wandb_history.jsonl` mirrors the values
sent to W&B, including when SDK logging is disabled. Final reports are W&B artifacts;
model/optimizer checkpoints remain local.

## Curriculum and controls

With U updates and R=max(2,ceil(U*ramp_fraction)), update index i starts at zero:

```text
gamma(i) = gamma_low + (gamma_high-gamma_low) * min(i/(R-1), 1)
```

Defaults are gamma 1..20, half the updates ramping and half at gamma 20.
`curriculum` uses this order; `hard` uses gamma 20 throughout; `shuffled` permutes
exactly the curriculum batch identities (gamma and seed, including within-batch
order). Shared cached labels make curriculum/shuffled examples identical. The
same model seed gives paired initialization. Hard-only matches updates/examples,
but can have different token counts and compute cost.

AdamW uses linear LR warmup followed by a **constant** LR. No decay. The ordinary
configuration retains the previous 24-layer, width-256, 4-head model, FF width 1024;
it is a configurable starting point, not a tuned numerical-integration recipe.

## Local smoke and reproduction

Use the repository virtualenv (Python 3.10+). Dependencies are listed in
`requirements.txt`; the existing environment already contains them.

```bash
.venv/bin/python -m unittest \
  experiments.integrals.tests.test_integrals \
  experiments.proposals.test_integral_ast_preview \
  experiments.proposals.test_integral_distribution -v

.venv/bin/python -m experiments.integrals.train \
  --config experiments/integrals/configs/smoke.json \
  --out-dir out/integrals-smoke --wandb offline
```

The smoke uses 2 layers, width 64, 4 heads, FF width 256, 128-token contexts,
4 examples/update and 8 updates on CPU. LR warms up for 2 updates, then stays at
0.0003. It evaluates two independent examples per split at each gamma 1..20 at
updates 0, 4, and 8; final evaluation uses another fresh two per split/gamma.
All evaluated examples also get free-generation checks in this small configuration.
It writes a real W&B **offline** run without uploading it. W&B still needs its
local service socket even in offline mode. `--wandb disabled` skips the SDK;
`--wandb online --wandb-project ... --wandb-entity ...` enables an authenticated run.

Outputs: `config.json`, `environment.json` (parameter count, versions, source
hashes), `vocabulary.json`, `metrics.jsonl`, `wandb_history.jsonl`, `final.json`, and
`checkpoint.pt` (model, optimizer, RNG, schedule position, counters).
The checkpoint is saved after each monitoring evaluation and at completion.

For interruption/resume, `--stop-after N` retains the original complete schedule;
`--resume --out-dir ...` loads the saved full configuration. Resume requires an
identical config/vocabulary and trusted local checkpoint. W&B online resumes use
the original destination and ID. Offline W&B runs cannot be appended reliably;
resume those with `--wandb disabled`. Local histories are truncated to the last
committed checkpoint on resume; already uploaded online events cannot be retracted.

Tests cover every canonical encoding and every positive midpoint, malformed labels,
numerical rejection/cache behavior, complete packing and prompt gradients, loss
decomposition, free decoding without answer leakage, independent evaluation banks,
exact shuffled batch matching, constant LR, and bit-identical CPU resume with dropout.

### Historical v1 smoke, 2026-10-25

[Saved verification report](results/smoke.json): **38 tests passed** (14 new and 24
generator/numerical regressions). The 109,568-parameter smoke model completed all
8 updates. All 240 monitoring loss series (20 gammas x 2 splits x 6 losses) were
verified at updates 0, 4, and 8 in the local history and in the real offline W&B
file, alongside a separate fresh final evaluation. The checkpoint reloaded with
strict parameter matching, and recorded source hashes identify the code executed
at that time. The v2 remote workflow has a separate validation report below.

The first pass generated 192 accepted labels, rejecting 12 candidates. The final
verification reused and revalidated that cache. On the fixed gamma-20 test samples,
answer NLL changed from 3.6345 to 3.2005 and full NLL from 3.7506 to 3.5558.
Free-generation validity remained zero: this is a successful pipeline smoke,
not evidence of learning integration. Two examples per evaluation cell are only
suitable for this smoke test.

Final artifacts are in `out/integrals-smoke-20261025-final/`, with checkpoint
`checkpoint.pt`, report `final.json`, and the real offline run under `wandb/`.

### Remote workflow validation

The [remote CPU validation report](results/remote_smoke.json) records parallel
label preparation, CPU-only batch/context calibration, and 56 updates for each
of curriculum, shuffled, and hard-only using immutable shards. It verifies exact
curriculum/shuffled batch identity, common initialization, all gamma 1..20 losses,
the separate decoding cadence, real offline W&B records, and durable snapshots.
This is a small pipeline check, not evidence of integration learning or an H100
throughput measurement. The remote recipe specifies the 201.5M-parameter model;
the local workflow check uses two layers and width 64.
