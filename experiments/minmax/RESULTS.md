# Initial CPU validation

Observed on 2026-10-17. These runs validate the implementation and provide early
learning measurements. They do not establish that curriculum wins or that the
smaller model has sufficient capacity to solve the task.

The PR validation on 2026-10-26 reran all 18 min/max tests, including benchmark
selection and bounded watchdog recovery, plus all 26 finite-field regression
tests. All 44 passed. The measurements below retain their original run dates,
configurations, and sample counts.

## Correctness and integration

- All 11 min/max tests pass. They cover all 41 values, every binary and ternary
  min/max input combination, negation, the independent AST/stack evaluators,
  sampling, unrestricted operator choice, exact shuffled examples, packing and
  causal answer masking, constant LR, evaluation independence, checkpoint resume,
  and comparison guards.
- All 26 existing finite-field tests still pass.
- A 109,888-parameter model memorized 16 fixed examples after 500 updates:
  loss 3.91581 to 0.00100, accuracy 100%. This is a training-path sanity check,
  not procedural generalization.
- The real W&B 0.30.0 SDK completed a six-update offline smoke run, with gamma
  train/test loss curves and a report artifact. Nothing was uploaded. Mocked W&B
  tests additionally verify preservation of exact CPU resume and run identity.
- Sampling and standalone checkpoint re-evaluation CLIs were exercised.

The validation record is [validation.json](results/validation.json).

## Tiny matched pilot

The [pilot recipe](configs/pilot.json) uses 2 layers, width 64, 4 heads, context
128, 109,888 parameters, 300 updates, batch 32, and seed 0. Each condition receives
9,600 supervised answers. Final measurements use 128 independent test items per
gamma, with shared samples across conditions. Curriculum and shuffled replay
identical generated batches in different orders.

| Condition | Gamma-12 test loss | Accuracy |
| --- | ---: | ---: |
| Curriculum | 3.65489 | 6.25% |
| Hard-only | 3.67585 | 5.47% |
| Shuffled | 3.65534 | 7.03% |

Uniform numeral guessing has loss 3.71357. The fitted label-frequency baseline
has loss 3.70392; the root-conditioned frequency baseline has loss 3.66383 on the
same final test sample. These results are close to simple distributional baselines
and do not demonstrate successful hierarchical evaluation or a curriculum benefit.

Raw paired summary: [pilot_summary.json](results/pilot_summary.json).
Full local reports/checkpoints: `out/minmax/pilot-seed0/<condition>-seed0/`.

## Small-model curriculum development run

The archived [4-layer recipe](configs/cpu_4layer.json) used **4 layers, width 128, 4 heads,
FF width 512, and 810,112 parameters**, including position embeddings. Tied
input/output embeddings use a 48-token vocabulary. Context is 128, batch size
is 64 answers, precision is FP32, and the device is CPU with four Torch threads.
The model is approximately 249 times smaller than the finite-field H100 model.
The main preset was subsequently changed to 24 layers and width 256 at the user's
request. The measurements below describe the original 4-layer run, not that new
18,931,968-parameter configuration.

Only the curriculum condition was run at this size: seed 0, 2,000 updates,
128,000 answers, gamma 1 to 12 over 800 updates then a plateau. Learning rate
warms up for 40 updates and stays at 0.0006. Training updates including generation
and packing took 50.79 seconds locally, excluding startup, evaluation, and
checkpoint writing. This is not a calibrated 50-minute ramp or a GPU benchmark.

Fresh final test banks have 512 items per gamma:

| Gamma | Model loss | Accuracy | Label-frequency loss | Root-conditioned frequency loss |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.29617 | 8.98% | 3.72507 | 3.60408 |
| 4 | 3.29133 | 11.72% | 3.72172 | 3.55564 |
| 8 | 3.36062 | 10.16% | 3.69980 | 3.50574 |
| 12 | 3.42530 | 10.55% | 3.69008 | 3.51319 |

There is modest learning, but accuracy remains low. Neither of the two fixed
structural pairs had both predictions correct. These observations do not establish
that the model learned the intended computation. Longer matched training and
capacity comparisons are still needed to assess the smaller-network hypothesis.
Do not compare this arm against the pilot controls as a curriculum effect: their
model size, batch size, training budget, and evaluation sample sizes differ.

All gamma train/test endpoints, baselines, configuration, and source/environment
metadata are in [small_curriculum.json](results/small_curriculum.json). The full
local output is `out/minmax/small-curriculum-seed0/`.

## Subsequent GPU experiment

The 24-layer, width-256 model completed training on an L4, with 18,964,736 parameters
at context 256. The revised 20-minute ramp + 20-minute plateau recipe uses
26,668 updates per condition and
256 supervised answers per update, with one seed for curriculum, hard, and
exact-batch shuffled controls. See [GPU_RUN.md](GPU_RUN.md) for timing,
configuration, monitoring, checkpoint locations, and W&B.

The [final gamma-12 comparison](results/gpu_gamma12_summary.json) uses 4,096
fresh test samples and one seed per condition:

| Condition | Test loss (nats) | Accuracy |
| --- | ---: | ---: |
| Curriculum | 0.132576 | 96.04% |
| Hard-only | 3.299532 | 8.23% |
| Exact-batch shuffled | 0.168483 | 95.04% |

Both curriculum and shuffled learn well on this seed; hard-only performs
poorly. Curriculum beats shuffled by 0.0359 nats and about one percentage point
of accuracy. This is a one-seed observation, not an estimate of variability
across seeds. The [gamma-20 repeat](GPU_GAMMA20.md) keeps the same update budget
and raises the maximum gamma. Its completed curriculum and hard-only arms reach
gamma-20 test losses of 0.183360 and 0.185806 respectively (95.09% and 94.75%
accuracy). Shuffled finished at loss 2.390637 and accuracy 32.96%, with the same
26,668 updates. The [final comparison](results/gpu_gamma20_summary.json) validates
matched configurations and fresh test banks. The [curve audit](GPU_GAMMA20_AUDIT.md)
documents the gradient instability observed in shuffled and independent
checkpoint checks. These one-seed results show earlier learning with curriculum
but almost equal final performance for curriculum and hard-only at gamma 20.

The [lower-LR repeat](GPU_GAMMA20_LOW_LR.md) halves the constant post-warmup LR
from 0.0006 to 0.0003, preserving the seed, model, generated batches, update
budget, and evaluation banks. Shuffled runs first. All three arms completed:

| Condition | Gamma-20 test loss | Accuracy |
| --- | ---: | ---: |
| Shuffled | 0.172622 | 95.95% |
| Curriculum | 0.131232 | 96.41% |
| Hard-only | 0.122971 | 96.83% |

The [final comparison](results/gpu_gamma20_low_lr_summary.json) uses 4,096 fresh
test samples per condition. Shuffled improves markedly from the previous run's
32.96% accuracy, supporting LR sensitivity as a contributor to the earlier
failure. This one-seed comparison does not establish a final-performance
advantage for curriculum over hard-only.
